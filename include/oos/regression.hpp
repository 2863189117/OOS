#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "oos/hdf5_io.hpp"
#include "oos/scene.hpp"
#include "oos/source.hpp"

namespace oos {

struct ResponseGrid {
  std::string domain_shape{"disk"};
  double radius_mm{};
  double half_x_mm{};
  double half_y_mm{};
  double line_y_start_mm{};
  double line_pitch_mm{};
  std::uint64_t line_count{};
  double spacing_mm{};
  std::uint64_t points{};
  std::uint64_t channels{};
  std::vector<double> xy_mm;
  std::vector<float> conditional_log_probability;
  std::vector<double> top_efficiency;
  std::vector<std::int32_t> channel_ids;
  std::string fingerprint_sha256;
  std::string effective_response_fingerprint_sha256;
  std::string effective_response_sha256;
  std::string source_angular_mode;
  std::string source_backend{"auto"};
  double source_z_mm{};
  double source_thickness_mm{};
  std::uint32_t source_transverse_count{};
  double obstacle_half_width_mm{};
  double obstacle_half_thickness_mm{};
  double source_medium_z_max_mm{};
  std::uint32_t source_mu_order{};
  std::uint32_t source_phi_count{};
  double source_relative_tolerance{1.0e-5};
  std::uint32_t source_maximum_subdivision_depth{8};
  std::uint32_t structured_disk_mu_order{31};
  std::uint32_t structured_disk_phi_count{64};

  void validate() const;
};

std::string response_grid_fingerprint(const ResponseGrid& grid);

struct RegressionResult {
  std::uint64_t events{};
  std::vector<double> fitted_xy_mm;
  std::vector<std::int32_t> fitted_line_id;
  std::vector<double> fitted_line_x_mm;
  std::vector<double> log_likelihood;
  std::vector<double> error_mm;
  std::vector<double> full_plane_log_likelihood;
  std::uint64_t likelihood_points{};
};

std::vector<double> cartesian_disk_grid(double radius_mm,
                                        double spacing_mm);
std::vector<double> cartesian_rectangle_grid(double half_x_mm,
                                             double half_y_mm,
                                             double spacing_mm);
std::vector<double> cartesian_parallel_line_grid(double half_x_mm, double spacing_mm,
                                        double line_y_start_mm,
                                        double line_pitch_mm,
                                        std::uint64_t line_count);

std::vector<SourceQuadrature> make_xy_shape_factor_sources(
    const std::vector<double>& xy_mm, std::int32_t domain,
    double z_min_mm = 0.0, double z_max_mm = 5.0,
    ShapeFactorOptions options = {});
std::vector<SourceQuadrature> make_xy_isotropic_product_sources(
    const std::vector<double>& xy_mm, std::int32_t domain, double z_mm,
    std::uint32_t mu_order, std::uint32_t phi_count);
std::vector<SourceQuadrature>
make_xy_rectangular_line_neighborhood_isotropic_product_sources(
    const std::vector<double>& xy_mm, std::int32_t domain,
    double line_center_z_mm, double obstacle_half_width_mm,
    double obstacle_half_thickness_mm, double maximum_distance_mm,
    double medium_z_max_mm, std::uint32_t transverse_count,
    std::uint32_t mu_order, std::uint32_t phi_count);

ResponseGrid make_response_grid(const std::vector<double>& xy_mm,
                                const std::vector<double>& efficiency,
                                std::uint64_t channels,
                                std::vector<std::int32_t> channel_ids,
                                double radius_mm, double spacing_mm,
                                double probability_floor = 1.0e-15,
                                std::string domain_shape = "disk",
                                double half_x_mm = 0.0,
                                double half_y_mm = 0.0,
                                double line_y_start_mm = 0.0,
                                double line_pitch_mm = 0.0,
                                std::uint64_t line_count = 0);

std::vector<double> score_response_grid_cpu(const ResponseGrid& grid,
                                            const HitBatch& hits);

RegressionResult fit_response_grid(const ResponseGrid& grid,
                                   const HitBatch& hits,
                                   bool retain_full_plane);
RegressionResult fit_response_grid_scores(
    const ResponseGrid& grid, const HitBatch& hits,
    std::vector<double> full_plane_log_likelihood, bool retain_full_plane);

std::vector<double> sensitive_channel_xy(const Scene& scene,
                                         const std::vector<std::int32_t>&
                                             channel_ids);

void save_response_grid_hdf5(const std::filesystem::path& path,
                             const ResponseGrid& grid);
ResponseGrid load_response_grid_hdf5(const std::filesystem::path& path);
void save_regression_hdf5(const std::filesystem::path& path,
                          const RegressionResult& result,
                          const std::vector<std::int32_t>& channel_ids);

}  // namespace oos
