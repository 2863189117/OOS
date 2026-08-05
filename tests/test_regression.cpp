#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "oos/regression.hpp"

TEST_CASE("Cartesian response grid is clipped to the active disk") {
  const auto xy = oos::cartesian_disk_grid(10.0, 10.0);
  REQUIRE(xy.size() == 10);
  for (std::size_t point = 0; point < xy.size() / 2; ++point)
    REQUIRE(std::hypot(xy[2 * point], xy[2 * point + 1]) <= 10.0);
}

TEST_CASE("finest sampled likelihood has a deterministic quadratic peak") {
  std::vector<double> xy;
  std::vector<double> score;
  for (int iy = -2; iy <= 2; ++iy)
    for (int ix = -2; ix <= 2; ++ix) {
      const double x = static_cast<double>(ix);
      const double y = static_cast<double>(iy);
      xy.push_back(x);
      xy.push_back(y);
      const double dx = x - 0.2;
      const double dy = y + 0.3;
      score.push_back(-(2.0 * dx * dx + dx * dy + 1.5 * dy * dy));
    }
  const auto best = static_cast<std::uint64_t>(
      std::max_element(score.begin(), score.end()) - score.begin());
  const auto peak =
      oos::fit_local_quadratic_peak(xy, score, best, 1.0, false);
  REQUIRE(peak.interpolated);
  REQUIRE(peak.x_mm == Catch::Approx(0.2).margin(1.0e-12));
  REQUIRE(peak.y_mm == Catch::Approx(-0.3).margin(1.0e-12));
  REQUIRE(peak.log_likelihood == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("non-concave finest likelihood falls back to its sampled maximum") {
  const std::vector<double> xy{-1.0, -1.0, 0.0, -1.0, 1.0, -1.0,
                               -1.0, 0.0,  0.0, 0.0,  1.0, 0.0,
                               -1.0, 1.0,  0.0, 1.0,  1.0, 1.0};
  std::vector<double> score;
  for (std::size_t point = 0; point < xy.size() / 2; ++point)
    score.push_back(xy[2 * point] * xy[2 * point] -
                    xy[2 * point + 1] * xy[2 * point + 1]);
  const auto peak = oos::fit_local_quadratic_peak(xy, score, 4, 1.0, false);
  REQUIRE_FALSE(peak.interpolated);
  REQUIRE(peak.x_mm == 0.0);
  REQUIRE(peak.y_mm == 0.0);
}

TEST_CASE("Cartesian response grid covers a rectangular source plane") {
  const auto xy = oos::cartesian_rectangle_grid(2.0, 1.0, 1.0);
  REQUIRE(xy.size() == 30);
  for (std::size_t point = 0; point < xy.size() / 2; ++point) {
    REQUIRE(std::abs(xy[2 * point]) <= 2.0);
    REQUIRE(std::abs(xy[2 * point + 1]) <= 1.0);
  }
}

TEST_CASE("Parallel-line response grid samples discrete parallel lines") {
  const auto xy = oos::cartesian_parallel_line_grid(2.0, 1.0, -0.5, 1.0, 2);
  REQUIRE(xy.size() == 20);
  for (std::size_t point = 0; point < xy.size() / 2; ++point) {
    REQUIRE(std::abs(xy[2 * point]) <= 2.0);
    REQUIRE((xy[2 * point + 1] == -0.5 ||
             xy[2 * point + 1] == 0.5));
  }
}

TEST_CASE("XY source candidates may lie on one fixed plane") {
  const auto sources =
      oos::make_xy_shape_factor_sources({1.0, -2.0}, 7, 17.49, 17.49);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].points.size() == 1);
  REQUIRE(sources[0].points[0].position.z == 17.49);
  REQUIRE(sources[0].points[0].domain == 7);
  REQUIRE(sources[0].points[0].stokes.i == 1.0);
}

TEST_CASE("XY isotropic-product sources are normalized on a fixed plane") {
  const auto sources =
      oos::make_xy_isotropic_product_sources({1.0, -2.0}, 7, 17.49, 4, 8);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].rays.size() == 32);
  double weight = 0.0;
  for (const auto& ray : sources[0].rays) {
    weight += ray.stokes.i;
    REQUIRE(ray.ray.origin.z == 17.49);
    REQUIRE(ray.domain == 7);
  }
  REQUIRE(weight == Catch::Approx(1.0));
}

TEST_CASE("Parallel-line sources integrate a bounded medium neighborhood") {
  const auto sources =
      oos::make_xy_rectangular_line_neighborhood_isotropic_product_sources(
          {1.0, -2.0}, 7, 17.499, 0.0025, 0.0005, 0.006, 17.5, 16, 4,
          8);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].rays.size() == 512);
  double weight = 0.0;
  for (const auto& ray : sources[0].rays) {
    weight += ray.stokes.i;
    REQUIRE(ray.ray.origin.y > -2.0085);
    REQUIRE(ray.ray.origin.y < -1.9915);
    REQUIRE(ray.ray.origin.z < 17.5);
    REQUIRE(ray.ray.origin.z > 17.4925);
    REQUIRE(ray.domain == 7);
  }
  REQUIRE(weight == Catch::Approx(1.0));
}

TEST_CASE("conditional grid likelihood selects the matching response") {
  const std::vector<double> xy{-1.0, 0.0, 1.0, 0.0};
  const std::vector<double> efficiency{0.18, 0.02, 0.02, 0.18};
  auto grid = oos::make_response_grid(xy, efficiency, 2, {10, 11}, 2.0,
                                      2.0, 1.0e-15, "rectangle", 1.0, 0.5);
  grid.effective_response_fingerprint_sha256 = "effective-fingerprint";
  grid.source_angular_mode = "shape_factor";
  grid.source_backend = "structured_analytic";
  grid.source_relative_tolerance = 2.0e-5;
  grid.source_maximum_subdivision_depth = 6;
  grid.structured_disk_mu_order = 31;
  grid.structured_disk_phi_count = 72;
  grid.fingerprint_sha256 = oos::response_grid_fingerprint(grid);
  oos::HitBatch hits;
  hits.count = 2;
  hits.channels = 2;
  hits.counts = {90, 10, 10, 90};
  hits.channel_ids = {10, 11};
  hits.truth_xy_mm = {-1.0, 0.0, 1.0, 0.0};
  const auto fit = oos::fit_response_grid(grid, hits, true);
  REQUIRE(fit.fitted_xy_mm == hits.truth_xy_mm);
  REQUIRE(fit.error_mm == std::vector<double>{0.0, 0.0});
  REQUIRE(fit.full_plane_log_likelihood.size() == 4);

  const auto path =
      std::filesystem::temp_directory_path() / "oos-response-grid-test.h5";
  oos::save_response_grid_hdf5(path, grid);
  const auto restored = oos::load_response_grid_hdf5(path);
  REQUIRE(restored.xy_mm == grid.xy_mm);
  REQUIRE(restored.domain_shape == grid.domain_shape);
  REQUIRE(restored.half_x_mm == grid.half_x_mm);
  REQUIRE(restored.half_y_mm == grid.half_y_mm);
  REQUIRE(restored.conditional_log_probability ==
          grid.conditional_log_probability);
  REQUIRE(restored.fingerprint_sha256 == grid.fingerprint_sha256);
  REQUIRE(restored.source_backend == "structured_analytic");
  REQUIRE(restored.source_relative_tolerance == 2.0e-5);
  REQUIRE(restored.source_maximum_subdivision_depth == 6);
  REQUIRE(restored.structured_disk_mu_order == 31);
  REQUIRE(restored.structured_disk_phi_count == 72);
  std::filesystem::remove(path);
}

TEST_CASE("regular-grid response interpolation scores sub-grid candidates") {
  const auto xy = oos::cartesian_rectangle_grid(1.0, 1.0, 1.0);
  std::vector<double> efficiency;
  for (std::size_t point = 0; point < xy.size() / 2; ++point) {
    const double fraction = 0.5 * (xy[2 * point] + 1.0);
    efficiency.push_back(0.18 - 0.16 * fraction);
    efficiency.push_back(0.02 + 0.16 * fraction);
  }
  const auto grid = oos::make_response_grid(
      xy, efficiency, 2, {10, 11}, std::sqrt(2.0), 1.0, 1.0e-15,
      "rectangle", 1.0, 1.0);
  oos::ResponseGridInterpolator interpolator(grid);
  const std::uint64_t right_counts[2]{10, 90};
  const double left = interpolator.score(-0.5, 0.25, right_counts);
  const double right = interpolator.score(0.5, 0.25, right_counts);
  REQUIRE(std::isfinite(left));
  REQUIRE(std::isfinite(right));
  REQUIRE(right > left);

  const std::uint64_t balanced_counts[2]{50, 50};
  const double conditional =
      interpolator.score(0.0, 0.0, balanced_counts);
  const double absolute =
      interpolator.score(0.0, 0.0, balanced_counts, 1000);
  REQUIRE(absolute < conditional);
}

TEST_CASE("Parallel-line response grid metadata round trips") {
  const std::vector<double> xy{-1.0, -0.5, 1.0, 0.5};
  const std::vector<double> efficiency{0.18, 0.02, 0.02, 0.18};
  auto grid = oos::make_response_grid(
      xy, efficiency, 2, {10, 11}, 2.0, 2.0, 1.0e-15, "parallel_lines",
      2.0, 0.5, -0.5, 1.0, 2);
  grid.source_angular_mode = "rectangular_line_neighborhood_isotropic_product";
  grid.effective_response_fingerprint_sha256 = "effective-fingerprint";
  grid.source_z_mm = 17.499;
  grid.source_thickness_mm = 0.006;
  grid.source_transverse_count = 64;
  grid.obstacle_half_width_mm = 0.0025;
  grid.obstacle_half_thickness_mm = 0.0005;
  grid.source_medium_z_max_mm = 17.5;
  grid.source_mu_order = 32;
  grid.source_phi_count = 128;
  grid.fingerprint_sha256 = oos::response_grid_fingerprint(grid);
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-line-response-grid-test.h5";
  oos::save_response_grid_hdf5(path, grid);
  const auto restored = oos::load_response_grid_hdf5(path);
  REQUIRE(restored.domain_shape == "parallel_lines");
  REQUIRE(restored.line_y_start_mm == -0.5);
  REQUIRE(restored.line_pitch_mm == 1.0);
  REQUIRE(restored.line_count == 2);
  REQUIRE(restored.source_angular_mode ==
          "rectangular_line_neighborhood_isotropic_product");
  REQUIRE(restored.source_z_mm == 17.499);
  REQUIRE(restored.source_thickness_mm == 0.006);
  REQUIRE(restored.source_transverse_count == 64);
  REQUIRE(restored.obstacle_half_width_mm == 0.0025);
  REQUIRE(restored.obstacle_half_thickness_mm == 0.0005);
  REQUIRE(restored.source_medium_z_max_mm == 17.5);
  REQUIRE(restored.source_mu_order == 32);
  REQUIRE(restored.source_phi_count == 128);
  REQUIRE(restored.fingerprint_sha256 == grid.fingerprint_sha256);
  oos::HitBatch hits;
  hits.count = 2;
  hits.channels = 2;
  hits.counts = {90, 10, 10, 90};
  hits.channel_ids = {10, 11};
  const auto fit = oos::fit_response_grid(restored, hits, false);
  REQUIRE(fit.fitted_line_id == std::vector<std::int32_t>{0, 1});
  REQUIRE(fit.fitted_line_x_mm == std::vector<double>{-1.0, 1.0});
  std::filesystem::remove(path);
}
