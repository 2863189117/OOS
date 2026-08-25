#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "oos/builder.hpp"

TEST_CASE("generic builder produces a conservative cavity operator") {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  scene.surfaces.emplace(
      0, oos::SurfaceModel{0, "wall", oos::SurfaceKind::lambertian, 0.8});
  oos::SurfaceModel detector;
  detector.id = 1;
  detector.name = "detector";
  detector.kind = oos::SurfaceKind::sensitive;
  detector.detection_probability = 1.0;
  detector.remainder = oos::RemainderAction::absorb;
  scene.surfaces.emplace(1, detector);
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id = {0, 1, 1, 1};
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, -1);
  scene.mesh.channel_id = {-1, 10, 11, 12};
  const auto operators = oos::OperatorBuilder::build(scene);
  REQUIRE(operators.transition.rows == 1);
  REQUIRE(operators.detection.cols == 3);
  double accounted = 0.0;
  for (double value : operators.transition.data) accounted += value;
  for (double value : operators.detection.data) accounted += value;
  for (double value : operators.losses.data) accounted += value;
  REQUIRE(accounted <= 1.0 + 1e-12);
  REQUIRE(accounted >= 1.0 - 1e-12);
}

TEST_CASE("surface basis groups fine geometry without losing conservation") {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  scene.surfaces.emplace(
      0, oos::SurfaceModel{0, "wall", oos::SurfaceKind::lambertian, 0.8});
  oos::SurfaceModel detector;
  detector.id = 1;
  detector.name = "detector";
  detector.kind = oos::SurfaceKind::sensitive;
  detector.detection_probability = 1.0;
  detector.remainder = oos::RemainderAction::absorb;
  scene.surfaces.emplace(1, detector);
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id = {0, 0, 1, 1};
  scene.mesh.surface_basis_id = {7, 7, 0, 1};
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, -1);
  scene.mesh.channel_id = {-1, -1, 10, 11};

  const auto grouped = oos::OperatorBuilder::build(scene);
  REQUIRE(grouped.transition.rows == 1);
  REQUIRE(grouped.state_labels == std::vector<std::string>{"wall/basis:7"});
  double accounted = 0.0;
  for (double value : grouped.transition.data) accounted += value;
  for (double value : grouped.detection.data) accounted += value;
  for (double value : grouped.losses.data) accounted += value;
  REQUIRE(std::abs(accounted - 1.0) <= 1e-12);

  scene.mesh.surface_basis_id.clear();
  const auto leaf = oos::OperatorBuilder::build(scene);
  REQUIRE(leaf.transition.rows == 2);
  REQUIRE(leaf.state_labels[0] == "wall/primitive:0");
  REQUIRE(leaf.state_labels[1] == "wall/primitive:1");
}

TEST_CASE("large-coordinate closed scenes use a float-safe ray offset") {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {1400.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  scene.surfaces.emplace(
      0, oos::SurfaceModel{0, "wall", oos::SurfaceKind::lambertian, 0.8});
  oos::SurfaceModel detector;
  detector.id = 1;
  detector.name = "detector";
  detector.kind = oos::SurfaceKind::sensitive;
  detector.detection_probability = 1.0;
  detector.remainder = oos::RemainderAction::absorb;
  scene.surfaces.emplace(1, detector);
  scene.mesh.vertices = {
      {1400, 0, 0}, {1401, 0, 0}, {1400, 1, 0}, {1400, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id = {0, 1, 1, 1};
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, -1);
  scene.mesh.channel_id = {-1, 10, 11, 12};
  const auto operators = oos::OperatorBuilder::build(scene);
  REQUIRE(operators.ray_origin_offset_mm >= 6.0e-3);
  const auto miss =
      std::find(operators.loss_names.begin(), operators.loss_names.end(),
                "float32_intersection_miss") -
      operators.loss_names.begin();
  for (std::uint64_t row = 0; row < operators.losses.rows; ++row)
    for (std::uint64_t entry = operators.losses.indptr[row];
         entry < operators.losses.indptr[row + 1]; ++entry)
      if (operators.losses.indices[entry] == miss)
        REQUIRE(operators.losses.data[entry] <= operators.tolerance);
}

TEST_CASE("exact solid-angle source projection closes a mixed cavity") {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(
      0, oos::Medium{0, "inside", 1.0,
                     std::numeric_limits<double>::infinity()});
  oos::SurfaceModel wall;
  wall.id = 0;
  wall.name = "wall";
  wall.kind = oos::SurfaceKind::lambertian;
  wall.reflectivity = 0.8;
  scene.surfaces.emplace(0, wall);
  oos::SurfaceModel detector;
  detector.id = 1;
  detector.name = "detector";
  detector.kind = oos::SurfaceKind::sensitive;
  detector.detection_probability = 1.0;
  detector.remainder = oos::RemainderAction::absorb;
  scene.surfaces.emplace(1, detector);
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id = {0, 1, 1, 1};
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, -1);
  scene.mesh.channel_id = {-1, 10, 10, 10};

  const auto operators = oos::OperatorBuilder::build(scene);
  oos::SourceQuadrature source;
  source.id = "shape-factor";
  source.integration =
      oos::SourceIntegration::isotropic_surface_shape_factor;
  source.points.push_back(
      {{0.1, 0.1, 0.1}, {1.0, 0.0, 0.0, 0.0}, 0});
  source.shape_factor.relative_tolerance = 1.0e-10;
  source.shape_factor.maximum_subdivision_depth = 4;
  source.shape_factor.maximum_approximate_solid_angle_fraction = 0.0;
  oos::SourceTraceRuntime runtime(scene, operators);
  const auto batch = runtime.trace({source});
  const auto reused_once = runtime.trace({source});
  const auto reused_twice = runtime.trace({source});
  CHECK(reused_once.initial_states == batch.initial_states);
  CHECK(reused_once.direct_detection == batch.direct_detection);
  CHECK(reused_once.direct_losses == batch.direct_losses);
  CHECK(reused_twice.initial_states == reused_once.initial_states);
  CHECK(reused_twice.direct_detection == reused_once.direct_detection);
  CHECK(reused_twice.direct_losses == reused_once.direct_losses);
  REQUIRE(batch.count == 1);
  REQUIRE(batch.source_integration_l1_error_estimate.size() == 1);
  CHECK(batch.source_integration_l1_error_estimate[0] >= 0.0);
  REQUIRE(batch.direct_detection.size() == 1);
  const double accounted =
      std::accumulate(batch.initial_states.begin(),
                      batch.initial_states.end(), 0.0) +
      std::accumulate(batch.direct_detection.begin(),
                      batch.direct_detection.end(), 0.0) +
      std::accumulate(batch.direct_losses.begin(), batch.direct_losses.end(),
                      0.0);
  CHECK(accounted == Catch::Approx(1.0).margin(1.0e-10));
  CHECK(batch.direct_losses.back() ==
        Catch::Approx(0.0).margin(1.0e-10));

  auto fallback = source;
  fallback.shape_factor.maximum_subdivision_depth = 0;
  fallback.shape_factor.maximum_approximate_solid_angle_fraction = 1.0e-3;
  auto exact_at_same_depth = fallback;
  exact_at_same_depth.shape_factor.maximum_approximate_solid_angle_fraction =
      0.0;
  const auto fallback_batch = runtime.trace({fallback});
  const auto exact_at_same_depth_batch = runtime.trace({exact_at_same_depth});
  CHECK(fallback_batch.initial_states ==
        exact_at_same_depth_batch.initial_states);
  CHECK(fallback_batch.direct_detection ==
        exact_at_same_depth_batch.direct_detection);
  CHECK(fallback_batch.direct_losses ==
        exact_at_same_depth_batch.direct_losses);

  auto approximate = source;
  approximate.shape_factor.maximum_approximate_solid_angle_fraction = 1.0;
  const auto approximate_batch = runtime.trace({approximate});
  REQUIRE(approximate_batch.initial_states.size() ==
          batch.initial_states.size());
  REQUIRE(approximate_batch.direct_detection.size() ==
          batch.direct_detection.size());
  for (std::size_t index = 0; index < batch.initial_states.size(); ++index)
    CHECK(approximate_batch.initial_states[index] ==
          Catch::Approx(batch.initial_states[index]).margin(1.0e-4));
  for (std::size_t index = 0; index < batch.direct_detection.size(); ++index)
    CHECK(approximate_batch.direct_detection[index] ==
          Catch::Approx(batch.direct_detection[index]).margin(1.0e-4));

  auto translated = source;
  translated.id = "shape-factor-translated";
  translated.points.front().position = {0.12, 0.11, 0.1};
  const auto batched = runtime.trace({source, translated});
  const auto translated_alone = runtime.trace({translated});
  REQUIRE(batched.count == 2);
  for (std::size_t column = 0; column < operators.transition.rows; ++column)
    CHECK(batched.initial_states[operators.transition.rows + column] ==
          Catch::Approx(translated_alone.initial_states[column])
              .margin(1.0e-13));
  for (std::size_t column = 0; column < operators.detection.cols; ++column)
    CHECK(batched.direct_detection[operators.detection.cols + column] ==
          Catch::Approx(translated_alone.direct_detection[column])
              .margin(1.0e-13));
  for (std::size_t column = 0; column < operators.losses.cols; ++column)
    CHECK(batched.direct_losses[operators.losses.cols + column] ==
          Catch::Approx(translated_alone.direct_losses[column])
              .margin(1.0e-13));
}

TEST_CASE("structured disk source integral replaces its validation fan") {
  constexpr std::uint32_t sectors = 256;
  constexpr double pi = 3.141592653589793238462643383279502884;
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.0, 0.0, 0.5};
  scene.media.emplace(
      0, oos::Medium{0, "inside", 1.0,
                     std::numeric_limits<double>::infinity()});
  oos::SurfaceModel detector;
  detector.id = 0;
  detector.name = "disk";
  detector.kind = oos::SurfaceKind::sensitive;
  detector.detection_probability = 1.0;
  detector.remainder = oos::RemainderAction::absorb;
  scene.surfaces.emplace(0, detector);
  oos::SurfaceModel wall;
  wall.id = 1;
  wall.name = "remainder";
  wall.kind = oos::SurfaceKind::lambertian;
  wall.reflectivity = 1.0;
  scene.surfaces.emplace(1, wall);

  scene.mesh.vertices.push_back({0.0, 0.0, 0.0});
  scene.mesh.vertices.push_back({0.0, 0.0, 1.0});
  for (std::uint32_t index = 0; index < sectors; ++index) {
    const double phi = 2.0 * pi * index / sectors;
    scene.mesh.vertices.push_back({std::cos(phi), std::sin(phi), 0.0});
    scene.mesh.vertices.push_back({std::cos(phi), std::sin(phi), 1.0});
  }
  const auto append = [&](std::array<std::uint32_t, 3> triangle,
                          std::uint32_t surface, std::int32_t channel,
                          std::uint8_t transport,
                          std::uint32_t replacement) {
    scene.mesh.triangles.push_back(triangle);
    scene.mesh.surface_id.push_back(surface);
    scene.mesh.surface_basis_id.push_back(0);
    scene.mesh.minus_domain_id.push_back(0);
    scene.mesh.plus_domain_id.push_back(-1);
    scene.mesh.channel_id.push_back(channel);
    scene.mesh.triangle_transport.push_back(transport);
    scene.mesh.triangle_source_quadrature.push_back(1);
    scene.mesh.triangle_source_analytic_primitive.push_back(replacement);
  };
  const auto missing = std::numeric_limits<std::uint32_t>::max();
  for (std::uint32_t index = 0; index < sectors; ++index) {
    const std::uint32_t next = (index + 1) % sectors;
    const std::uint32_t bottom = 2 + 2 * index;
    const std::uint32_t bottom_next = 2 + 2 * next;
    const std::uint32_t top = bottom + 1;
    const std::uint32_t top_next = bottom_next + 1;
    append({0, bottom_next, bottom}, 0, 10, 0, 0);
    append({1, top, top_next}, 1, -1, 1, missing);
    append({bottom, bottom_next, top_next}, 1, -1, 1, missing);
    append({bottom, top_next, top}, 1, -1, 1, missing);
  }
  oos::AnalyticPrimitive disk;
  disk.kind = oos::GeometryPrimitiveKind::disk;
  disk.center_mm = {0.0, 0.0, 0.0};
  disk.parameters = {1.0, 0.0, 0.0, 0.0};
  disk.normal_sign = -1.0;
  disk.surface_id = 0;
  disk.minus_domain_id = 0;
  disk.plus_domain_id = -1;
  disk.channel_id = 10;
  disk.source_integral = oos::AnalyticSourceIntegral::directional_disk;
  scene.mesh.analytic_primitives.push_back(disk);

  const auto operators = oos::OperatorBuilder::build(scene);
  oos::SourceQuadrature source;
  source.id = "structured-disk";
  source.integration =
      oos::SourceIntegration::isotropic_surface_shape_factor;
  source.points.push_back(
      {{0.0, 0.0, 0.5}, {1.0, 0.0, 0.0, 0.0}, 0});
  source.shape_factor.backend = oos::ShapeFactorBackend::automatic;
  source.shape_factor.maximum_subdivision_depth = 0;
  const oos::SourceTraceRuntime runtime(scene, operators);
  const auto batch = runtime.trace({source});
  const auto disk_column =
      std::find(operators.channel_ids.begin(), operators.channel_ids.end(),
                10) -
      operators.channel_ids.begin();
  REQUIRE(disk_column < operators.channel_ids.size());
  const double expected = 0.5 * (1.0 - 0.5 / std::hypot(0.5, 1.0));
  CHECK(batch.direct_detection[disk_column] ==
        Catch::Approx(expected).margin(2.0e-4));
  const double accounted =
      std::accumulate(batch.initial_states.begin(),
                      batch.initial_states.end(), 0.0) +
      std::accumulate(batch.direct_detection.begin(),
                      batch.direct_detection.end(), 0.0) +
      std::accumulate(batch.direct_losses.begin(), batch.direct_losses.end(),
                      0.0);
  CHECK(accounted ==
        Catch::Approx(1.0).margin(1.0e-10));
  CHECK(batch.source_integration_l1_error_estimate.front() < 2.0e-4);
}
