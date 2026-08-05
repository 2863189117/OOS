#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oos/cuda_solver.hpp"
#include "oos/effective_response.hpp"
#include "oos/hash.hpp"
#include "oos/hdf5_io.hpp"
#include "oos/regression.hpp"
#include "oos/scene.hpp"
#include "oos/source.hpp"

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  oos-regress grid operators.h5 --precomputed effective.h5 "
         "--scene scene.yaml --output grid.h5 [--spacing-mm 10] "
         "[--grid-shape disk|rectangle|parallel_lines] [--radius-mm 1000] "
         "[--half-x-mm 34] [--half-y-mm 34] "
         "[--line-y-start-mm -33.5] [--line-pitch-mm 1] "
         "[--line-count 68] "
         "[--source-z-mm 2.5] [--source-thickness-mm 5] "
         "[--source-angular-mode "
         "shape_factor|isotropic_product|rectangular_line_neighborhood_isotropic_product] "
         "[--source-transverse-count 32] "
         "[--obstacle-half-width-mm 0.0025] "
         "[--obstacle-half-thickness-mm 0.0005] "
         "[--source-medium-z-max-mm 17.5] "
         "[--source-mu-order 32] [--source-phi-count 128] "
         "[--source-backend auto|generic_bvh|structured_analytic] "
         "[--structured-disk-mu-order 31] "
         "[--structured-disk-phi-count 64] "
         "[--source-subdivision-depth 8] [--batch-size auto|N] "
         "[--verify-effective-content] "
         "[--surface-basis selection.h5] [--device cpu|cuda]\n"
      << "  oos-regress fit --hits hits.h5 --output regression.h5 "
         "[--fit-mode fast|accurate] [--grid grid.h5] "
         "[--coarse-spacing-mm 50] [--operators operators.h5 "
         "--precomputed effective.h5 --scene scene.yaml] "
         "[--surface-basis selection.h5] [--device cpu|cuda] "
         "[--source-z-mm 2.5] [--source-thickness-mm 5] "
         "[--source-angular-mode "
         "shape_factor|isotropic_product|rectangular_line_neighborhood_isotropic_product] "
         "[--source-transverse-count 32] "
         "[--source-mu-order 32] [--source-phi-count 128] "
         "[--source-backend auto|generic_bvh|structured_analytic] "
         "[--structured-disk-mu-order 31] "
         "[--structured-disk-phi-count 64] "
         "[--refine-spacings-mm LIST] "
         "[--refine-half-widths-mm LIST] "
         "[--batch-size auto|N] "
         "[--adjoint]\n";
}

std::string option(int argc, char** argv, const std::string& name,
                   const std::string& fallback = "") {
  for (int i = 0; i + 1 < argc; ++i)
    if (argv[i] == name) return argv[i + 1];
  return fallback;
}

bool flag(int argc, char** argv, const std::string& name) {
  for (int i = 0; i < argc; ++i)
    if (argv[i] == name) return true;
  return false;
}

oos::ShapeFactorBackend shape_factor_backend(const std::string& value) {
  if (value == "auto") return oos::ShapeFactorBackend::automatic;
  if (value == "generic_bvh") return oos::ShapeFactorBackend::generic_bvh;
  if (value == "structured_analytic")
    return oos::ShapeFactorBackend::structured_analytic;
  throw std::runtime_error(
      "--source-backend must be auto, generic_bvh, or structured_analytic");
}

std::string shape_factor_backend_name(oos::ShapeFactorBackend backend) {
  switch (backend) {
    case oos::ShapeFactorBackend::automatic:
      return "auto";
    case oos::ShapeFactorBackend::generic_bvh:
      return "generic_bvh";
    case oos::ShapeFactorBackend::structured_analytic:
      return "structured_analytic";
  }
  throw std::runtime_error("unknown shape-factor backend");
}

std::uint64_t source_batch_size(
    const std::string& value, const oos::EffectiveResponse& response) {
  if (value != "auto") {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0)
      throw std::runtime_error(
          "--batch-size must be auto or a positive integer");
    return parsed;
  }
  std::uint64_t workers = 1;
#ifdef _OPENMP
  workers = static_cast<std::uint64_t>(omp_get_max_threads());
#endif
  constexpr std::uint64_t memory_cap = 512ull * 1024ull * 1024ull;
  const std::uint64_t values_per_candidate =
      response.states + 2 * response.channels + response.losses;
  const std::uint64_t bytes_per_candidate =
      std::max<std::uint64_t>(1, values_per_candidate) * sizeof(double);
  const std::uint64_t memory_limited =
      std::max<std::uint64_t>(1, memory_cap / bytes_per_candidate);
  return std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(4 * workers, memory_limited));
}

std::vector<double> comma_separated_doubles(const std::string& value) {
  std::vector<double> result;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (field.empty())
      throw std::runtime_error("empty value in comma-separated option");
    const double parsed = std::stod(field);
    if (!std::isfinite(parsed) || parsed <= 0.0)
      throw std::runtime_error(
          "refinement values must be finite and positive");
    result.push_back(parsed);
  }
  if (result.empty())
    throw std::runtime_error("comma-separated option must not be empty");
  return result;
}

struct SearchDomain {
  std::string shape{"disk"};
  double radius{1000.0};
  double half_x{};
  double half_y{};
  double line_y_start{};
  double line_pitch{};
  std::uint64_t line_count{};

  bool contains(double x, double y) const {
    if (shape == "disk")
      return x * x + y * y <= radius * radius + 1.0e-9;
    if (shape == "rectangle")
      return std::abs(x) <= half_x + 1.0e-9 &&
             std::abs(y) <= half_y + 1.0e-9;
    if (std::abs(x) > half_x + 1.0e-9 || line_pitch <= 0.0 ||
        line_count == 0)
      return false;
    const double coordinate = (y - line_y_start) / line_pitch;
    const auto line = static_cast<std::int64_t>(std::llround(coordinate));
    return line >= 0 &&
           line < static_cast<std::int64_t>(line_count) &&
           std::abs(coordinate - line) <= 1.0e-9;
  }
};

SearchDomain search_domain_from_arguments(int argc, char** argv) {
  SearchDomain result;
  result.shape = option(argc, argv, "--grid-shape", "disk");
  result.radius = std::stod(option(argc, argv, "--radius-mm", "1000"));
  if (result.shape == "rectangle") {
    result.half_x = std::stod(option(argc, argv, "--half-x-mm", "34"));
    result.half_y = std::stod(option(argc, argv, "--half-y-mm", "34"));
    if (!(result.half_x > 0.0) || !(result.half_y > 0.0))
      throw std::runtime_error(
          "rectangle half-widths must be positive");
    result.radius = std::hypot(result.half_x, result.half_y);
  } else if (result.shape == "parallel_lines") {
    result.half_x =
        std::stod(option(argc, argv, "--half-x-mm", "34"));
    result.line_y_start =
        std::stod(option(argc, argv, "--line-y-start-mm", "-33.5"));
    result.line_pitch =
        std::stod(option(argc, argv, "--line-pitch-mm", "1"));
    result.line_count = static_cast<std::uint64_t>(
        std::stoull(option(argc, argv, "--line-count", "68")));
    if (!(result.half_x > 0.0) || !std::isfinite(result.line_y_start) ||
        !(result.line_pitch > 0.0) || result.line_count == 0)
      throw std::runtime_error("line-line grid parameters are invalid");
    const double last_y =
        result.line_y_start + result.line_pitch * (result.line_count - 1);
    result.half_y =
        std::max(std::abs(result.line_y_start), std::abs(last_y));
    result.radius = std::hypot(result.half_x, result.half_y);
  } else if (result.shape != "disk") {
    throw std::runtime_error(
        "--grid-shape must be disk, rectangle, or parallel_lines");
  }
  if (!(result.radius > 0.0))
    throw std::runtime_error("grid radius must be positive");
  return result;
}

SearchDomain search_domain_from_grid(const oos::ResponseGrid& grid) {
  return {grid.domain_shape,
          grid.radius_mm,
          grid.half_x_mm,
          grid.half_y_mm,
          grid.line_y_start_mm,
          grid.line_pitch_mm,
          grid.line_count};
}

std::vector<std::pair<double, double>> refinement_levels(
    int argc, char** argv, bool fast_mode) {
  const auto spacing_option = option(argc, argv, "--refine-spacings-mm");
  const auto width_option = option(argc, argv, "--refine-half-widths-mm");
  if (spacing_option.empty() != width_option.empty())
    throw std::runtime_error(
        "both refinement spacing and half-width options are required");
  if (spacing_option.empty() && fast_mode)
    return {{10.0, 80.0}, {5.0, 10.0}, {1.0, 2.0},
            {0.25, 0.5}};
  if (spacing_option.empty())
    return {{40.0, 160.0}, {5.0, 20.0}, {1.0, 4.0},
            {0.25, 1.0}, {0.1, 0.4}};
  const auto spacings = comma_separated_doubles(spacing_option);
  const auto widths = comma_separated_doubles(width_option);
  if (spacings.size() != widths.size())
    throw std::runtime_error(
        "refinement spacing and half-width counts differ");
  std::vector<std::pair<double, double>> result;
  result.reserve(spacings.size());
  for (std::size_t index = 0; index < spacings.size(); ++index)
    result.emplace_back(spacings[index], widths[index]);
  return result;
}

oos::ResponseGrid coarse_response_grid(const oos::ResponseGrid& grid,
                                       double coarse_spacing_mm) {
  grid.validate();
  const double ratio = coarse_spacing_mm / grid.spacing_mm;
  const auto stride = static_cast<std::int64_t>(std::llround(ratio));
  if (stride < 1 || std::abs(ratio - stride) > 1.0e-9)
    throw std::runtime_error(
        "--coarse-spacing-mm must be an integer multiple of grid spacing");
  std::vector<std::uint64_t> selected;
  selected.reserve(grid.points / static_cast<std::uint64_t>(stride * stride) +
                   1024);
  for (std::uint64_t point = 0; point < grid.points; ++point) {
    const double x = grid.xy_mm[2 * point];
    const double y = grid.xy_mm[2 * point + 1];
    const auto ix = static_cast<std::int64_t>(
        std::llround(x / grid.spacing_mm));
    bool coarse = ix % stride == 0;
    bool boundary = false;
    if (grid.domain_shape == "parallel_lines") {
      boundary = std::abs(x) >= grid.half_x_mm - 1.5 * grid.spacing_mm;
    } else {
      const auto iy = static_cast<std::int64_t>(
          std::llround(y / grid.spacing_mm));
      coarse = coarse && iy % stride == 0;
      if (grid.domain_shape == "disk")
        boundary = std::hypot(x, y) >=
                   grid.radius_mm - 1.5 * grid.spacing_mm;
      else
        boundary =
            std::abs(x) >= grid.half_x_mm - 1.5 * grid.spacing_mm ||
            std::abs(y) >= grid.half_y_mm - 1.5 * grid.spacing_mm;
    }
    if (coarse || boundary) selected.push_back(point);
  }
  oos::ResponseGrid result = grid;
  result.points = selected.size();
  result.xy_mm.clear();
  result.top_efficiency.clear();
  result.conditional_log_probability.clear();
  result.xy_mm.reserve(result.points * 2);
  result.top_efficiency.reserve(result.points);
  result.conditional_log_probability.reserve(result.points * grid.channels);
  for (const auto point : selected) {
    result.xy_mm.push_back(grid.xy_mm[2 * point]);
    result.xy_mm.push_back(grid.xy_mm[2 * point + 1]);
    result.top_efficiency.push_back(grid.top_efficiency[point]);
    const auto begin =
        grid.conditional_log_probability.begin() + point * grid.channels;
    result.conditional_log_probability.insert(
        result.conditional_log_probability.end(), begin,
        begin + grid.channels);
  }
  result.fingerprint_sha256.clear();
  result.validate();
  return result;
}

oos::Scene configured_scene(int argc, char** argv,
                            const std::filesystem::path& path) {
  auto scene = oos::Scene::load(path);
  const auto basis = option(argc, argv, "--surface-basis");
  if (!basis.empty()) scene.apply_surface_basis(basis);
  return scene;
}

std::vector<double> local_grid(double center_x, double center_y,
                               double spacing, double half_width,
                               const SearchDomain& domain) {
  const auto count =
      static_cast<std::int64_t>(std::floor(half_width / spacing));
  std::vector<double> points;
  if (domain.shape == "parallel_lines") {
    const auto first_line = std::max<std::int64_t>(
        0, static_cast<std::int64_t>(std::ceil(
               (center_y - half_width - domain.line_y_start) /
               domain.line_pitch)));
    const auto last_line = std::min<std::int64_t>(
        static_cast<std::int64_t>(domain.line_count) - 1,
        static_cast<std::int64_t>(std::floor(
            (center_y + half_width - domain.line_y_start) /
            domain.line_pitch)));
    for (std::int64_t line = first_line; line <= last_line; ++line) {
      const double y = domain.line_y_start + line * domain.line_pitch;
      for (std::int64_t ix = -count; ix <= count; ++ix) {
        const double x = center_x + spacing * ix;
        if (domain.contains(x, y)) {
          points.push_back(x);
          points.push_back(y);
        }
      }
    }
    return points;
  }
  for (std::int64_t iy = -count; iy <= count; ++iy)
    for (std::int64_t ix = -count; ix <= count; ++ix) {
      const double x = center_x + spacing * ix;
      const double y = center_y + spacing * iy;
      if (domain.contains(x, y)) {
        points.push_back(x);
        points.push_back(y);
      }
    }
  return points;
}

class CandidateEvaluator {
 public:
  CandidateEvaluator(oos::Scene scene, oos::OperatorSet operators,
                     oos::EffectiveResponse response, std::string device,
                     double source_z_mm, double source_thickness_mm,
                     oos::ShapeFactorOptions shape_factor,
                     std::string source_angular_mode,
                     std::uint32_t source_transverse_count,
                     double obstacle_half_width_mm,
                     double obstacle_half_thickness_mm,
                     double source_medium_z_max_mm,
                     std::uint32_t source_mu_order,
                     std::uint32_t source_phi_count)
      : scene_(std::move(scene)),
        operators_(std::move(operators)),
        response_(std::move(response)),
        device_(std::move(device)),
        source_z_mm_(source_z_mm),
        source_thickness_mm_(source_thickness_mm),
        shape_factor_(shape_factor),
        source_angular_mode_(std::move(source_angular_mode)),
        source_transverse_count_(source_transverse_count),
        obstacle_half_width_mm_(obstacle_half_width_mm),
        obstacle_half_thickness_mm_(obstacle_half_thickness_mm),
        source_medium_z_max_mm_(source_medium_z_max_mm),
        source_mu_order_(source_mu_order),
        source_phi_count_(source_phi_count) {
    if (response_.operator_cache_key_sha256 !=
        operators_.cache_key_sha256)
      throw std::runtime_error(
          "precomputed response does not match operators.h5");
    source_runtime_ =
        std::make_unique<oos::SourceTraceRuntime>(scene_, operators_);
#ifdef OOS_HAS_CUDA
    if (device_ == "cuda")
      cuda_ =
          std::make_unique<oos::CudaEffectiveResponseRuntime>(response_);
#endif
    if (device_ != "cpu" && device_ != "cuda")
      throw std::runtime_error("--device must be cpu or cuda");
  }

  oos::SolveResult evaluate(const std::vector<double>& xy_mm) {
    std::vector<oos::SourceQuadrature> quadratures;
    if (source_angular_mode_ == "isotropic_product") {
      if (source_thickness_mm_ != 0.0)
        throw std::runtime_error(
            "isotropic-product regression requires a fixed source plane");
      quadratures = oos::make_xy_isotropic_product_sources(
          xy_mm, scene_.primary_domain, source_z_mm_, source_mu_order_,
          source_phi_count_);
    } else if (source_angular_mode_ ==
               "rectangular_line_neighborhood_isotropic_product") {
      quadratures =
          oos::make_xy_rectangular_line_neighborhood_isotropic_product_sources(
              xy_mm, scene_.primary_domain, source_z_mm_,
              obstacle_half_width_mm_, obstacle_half_thickness_mm_,
              source_thickness_mm_, source_medium_z_max_mm_,
              source_transverse_count_, source_mu_order_,
              source_phi_count_);
    } else if (source_angular_mode_ == "shape_factor") {
      const double half_thickness = 0.5 * source_thickness_mm_;
      quadratures = oos::make_xy_shape_factor_sources(
          xy_mm, scene_.primary_domain, source_z_mm_ - half_thickness,
          source_z_mm_ + half_thickness, shape_factor_);
    } else {
      throw std::runtime_error(
          "--source-angular-mode must be shape_factor, isotropic_product, "
          "or rectangular_line_neighborhood_isotropic_product");
    }
    const auto sources = source_runtime_->trace(quadratures);
    if (device_ == "cpu")
      return oos::apply_effective_response_cpu(response_, sources);
#ifdef OOS_HAS_CUDA
    return cuda_->apply(sources);
#else
    throw std::runtime_error("this build has no CUDA backend");
#endif
  }

  const oos::EffectiveResponse& response() const { return response_; }

 private:
  oos::Scene scene_;
  oos::OperatorSet operators_;
  oos::EffectiveResponse response_;
  std::string device_;
  double source_z_mm_{};
  double source_thickness_mm_{};
  oos::ShapeFactorOptions shape_factor_;
  std::string source_angular_mode_;
  std::uint32_t source_transverse_count_{};
  double obstacle_half_width_mm_{};
  double obstacle_half_thickness_mm_{};
  double source_medium_z_max_mm_{};
  std::uint32_t source_mu_order_{};
  std::uint32_t source_phi_count_{};
  std::unique_ptr<oos::SourceTraceRuntime> source_runtime_;
#ifdef OOS_HAS_CUDA
  std::unique_ptr<oos::CudaEffectiveResponseRuntime> cuda_;
#endif
};

double observation_score(const double* efficiency, std::uint64_t channels,
                         const std::uint64_t* counts,
                         std::uint64_t emitted) {
  const double total =
      std::accumulate(efficiency, efficiency + channels, 0.0);
  if (!(total > 0.0) || !(total < 1.0))
    return -std::numeric_limits<double>::infinity();
  double score = 0.0;
  std::uint64_t top_hits = 0;
  for (std::uint64_t channel = 0; channel < channels; ++channel) {
    top_hits += counts[channel];
    const double probability =
        emitted == 0 ? efficiency[channel] / total : efficiency[channel];
    score += static_cast<double>(counts[channel]) *
             std::log(std::max(probability, 1.0e-15));
  }
  if (emitted != 0) {
    if (emitted < top_hits)
      throw std::runtime_error(
          "emitted count is smaller than observed top hits");
    score += static_cast<double>(emitted - top_hits) *
             std::log(std::max(1.0 - total, 1.0e-15));
  }
  return score;
}

struct CandidateMaximum {
  std::vector<double> xy_mm;
  double score{-std::numeric_limits<double>::infinity()};
  std::uint64_t best_index{};
  std::vector<double> sampled_scores;
};

CandidateMaximum maximize_candidates(
    CandidateEvaluator& evaluator, const std::vector<double>& xy_mm,
    const std::uint64_t* counts, std::uint64_t channels,
    std::uint64_t emitted, std::uint64_t batch_size, bool retain_scores) {
  const auto points = static_cast<std::uint64_t>(xy_mm.size() / 2);
  if (batch_size == 0)
    throw std::runtime_error("--batch-size must be positive");
  CandidateMaximum result;
  if (retain_scores)
    result.sampled_scores.assign(
        points, -std::numeric_limits<double>::infinity());
  for (std::uint64_t start = 0; start < points; start += batch_size) {
    const auto stop = std::min(points, start + batch_size);
    const std::vector<double> batch(xy_mm.begin() + 2 * start,
                                    xy_mm.begin() + 2 * stop);
    const auto response = evaluator.evaluate(batch);
    for (std::uint64_t point = start; point < stop; ++point) {
      const auto local = point - start;
      const double score = observation_score(
          response.efficiency.data() + local * channels, channels, counts,
          emitted);
      if (retain_scores) result.sampled_scores[point] = score;
      if (score > result.score) {
        result.score = score;
        result.best_index = point;
      }
    }
  }
  if (!std::isfinite(result.score))
    throw std::runtime_error("all matrix refinement candidates were invalid");
  result.xy_mm = {xy_mm[2 * result.best_index],
                  xy_mm[2 * result.best_index + 1]};
  return result;
}

void refine_events_accurate(
    CandidateEvaluator& evaluator, const oos::HitBatch& hits,
    const SearchDomain& domain,
    const std::vector<std::pair<double, double>>& levels,
    bool preserve_large_disk_edge_search, std::uint64_t batch_size,
    oos::RegressionResult& result) {
  if (levels.empty()) throw std::runtime_error("refinement levels are empty");
  result.subgrid_interpolated.assign(hits.count, 0);
  result.fit_mode = "accurate_matrix";
  result.likelihood = hits.emitted.empty() ? "conditional_multinomial"
                                           : "absolute_multinomial";
  result.final_sampling_spacing_mm = levels.back().first;
  for (std::uint64_t event = 0; event < hits.count; ++event) {
    double center_x = result.fitted_xy_mm[2 * event];
    double center_y = result.fitted_xy_mm[2 * event + 1];
    double score = result.log_likelihood[event];
    auto event_levels = levels;
    if (preserve_large_disk_edge_search && domain.shape == "disk" &&
        std::hypot(center_x, center_y) >= 850.0 &&
        !event_levels.empty())
      event_levels.front().second = 400.0;
    for (std::size_t level = 0; level < event_levels.size(); ++level) {
      const auto [spacing, half_width] = event_levels[level];
      auto points =
          local_grid(center_x, center_y, spacing, half_width, domain);
      if (points.empty())
        throw std::runtime_error(
            "continuous refinement produced no in-domain candidates");
      const bool final_level = level + 1 == event_levels.size();
      const auto fitted = maximize_candidates(
          evaluator, points, hits.counts.data() + event * hits.channels,
          hits.channels,
          hits.emitted.empty() ? 0 : hits.emitted[event], batch_size,
          final_level);
      center_x = fitted.xy_mm[0];
      center_y = fitted.xy_mm[1];
      score = fitted.score;
      if (final_level) {
        const auto peak = oos::fit_local_quadratic_peak(
            points, fitted.sampled_scores, fitted.best_index, spacing,
            domain.shape == "parallel_lines");
        if (peak.interpolated && domain.contains(peak.x_mm, peak.y_mm)) {
          center_x = peak.x_mm;
          center_y = peak.y_mm;
          score = peak.log_likelihood;
          result.subgrid_interpolated[event] = 1;
        }
      }
    }
    result.fitted_xy_mm[2 * event] = center_x;
    result.fitted_xy_mm[2 * event + 1] = center_y;
    if (!result.fitted_line_id.empty()) {
      result.fitted_line_id[event] =
          static_cast<std::int32_t>(std::llround(
              (center_y - domain.line_y_start) / domain.line_pitch));
      result.fitted_line_x_mm[event] = center_x;
    }
    result.log_likelihood[event] = score;
    if (!hits.truth_xy_mm.empty()) {
      const double dx = center_x - hits.truth_xy_mm[2 * event];
      const double dy = center_y - hits.truth_xy_mm[2 * event + 1];
      result.error_mm[event] = std::hypot(dx, dy);
    }
  }
}

void refine_events_fast(
    const oos::ResponseGrid& grid, const oos::HitBatch& hits,
    const SearchDomain& domain,
    const std::vector<std::pair<double, double>>& levels,
    oos::RegressionResult& result) {
  if (levels.empty()) throw std::runtime_error("refinement levels are empty");
  oos::ResponseGridInterpolator interpolator(grid);
  result.subgrid_interpolated.assign(hits.count, 0);
  result.fit_mode = "fast_grid_interpolated";
  result.likelihood = hits.emitted.empty() ? "conditional_multinomial"
                                           : "absolute_multinomial";
  result.final_sampling_spacing_mm = levels.back().first;
  std::atomic<bool> invalid_candidates{false};
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t signed_event = 0;
       signed_event < static_cast<std::int64_t>(hits.count);
       ++signed_event) {
    const auto event = static_cast<std::uint64_t>(signed_event);
    if (invalid_candidates.load(std::memory_order_relaxed)) continue;
    double center_x = result.fitted_xy_mm[2 * event];
    double center_y = result.fitted_xy_mm[2 * event + 1];
    double score = result.log_likelihood[event];
    for (std::size_t level = 0; level < levels.size(); ++level) {
      const auto [spacing, half_width] = levels[level];
      const auto points =
          local_grid(center_x, center_y, spacing, half_width, domain);
      if (points.empty()) {
        invalid_candidates.store(true, std::memory_order_relaxed);
        break;
      }
      const auto point_count = points.size() / 2;
      std::vector<double> sampled(
          point_count, -std::numeric_limits<double>::infinity());
      std::uint64_t best = 0;
      double best_score = -std::numeric_limits<double>::infinity();
      for (std::uint64_t point = 0; point < point_count; ++point) {
        sampled[point] = interpolator.score(
            points[2 * point], points[2 * point + 1],
            hits.counts.data() + event * hits.channels,
            hits.emitted.empty() ? 0 : hits.emitted[event]);
        if (sampled[point] > best_score) {
          best_score = sampled[point];
          best = point;
        }
      }
      if (!std::isfinite(best_score)) {
        invalid_candidates.store(true, std::memory_order_relaxed);
        break;
      }
      center_x = points[2 * best];
      center_y = points[2 * best + 1];
      score = best_score;
      if (level + 1 == levels.size()) {
        const auto peak = oos::fit_local_quadratic_peak(
            points, sampled, best, spacing,
            domain.shape == "parallel_lines");
        if (peak.interpolated && domain.contains(peak.x_mm, peak.y_mm)) {
          center_x = peak.x_mm;
          center_y = peak.y_mm;
          score = peak.log_likelihood;
          result.subgrid_interpolated[event] = 1;
        }
      }
    }
    result.fitted_xy_mm[2 * event] = center_x;
    result.fitted_xy_mm[2 * event + 1] = center_y;
    if (!result.fitted_line_id.empty()) {
      result.fitted_line_id[event] =
          static_cast<std::int32_t>(std::llround(
              (center_y - domain.line_y_start) / domain.line_pitch));
      result.fitted_line_x_mm[event] = center_x;
    }
    result.log_likelihood[event] = score;
    if (!hits.truth_xy_mm.empty()) {
      const double dx = center_x - hits.truth_xy_mm[2 * event];
      const double dy = center_y - hits.truth_xy_mm[2 * event + 1];
      result.error_mm[event] = std::hypot(dx, dy);
    }
  }
  if (invalid_candidates.load())
    throw std::runtime_error(
        "fast refinement encountered candidates outside grid support");
}

oos::RegressionResult centroid_initial(const oos::Scene& scene,
                                       const oos::EffectiveResponse& response,
                                       const oos::HitBatch& hits,
                                       const SearchDomain& domain) {
  const auto channel_xy = oos::sensitive_channel_xy(scene, response.channel_ids);
  oos::RegressionResult result;
  result.events = hits.count;
  result.fitted_xy_mm.resize(hits.count * 2);
  if (domain.shape == "parallel_lines") {
    result.fitted_line_id.resize(hits.count);
    result.fitted_line_x_mm.resize(hits.count);
  }
  result.log_likelihood.assign(
      hits.count, -std::numeric_limits<double>::infinity());
  if (!hits.truth_xy_mm.empty()) result.error_mm.resize(hits.count);
  for (std::uint64_t event = 0; event < hits.count; ++event) {
    double total = 0.0;
    double x = 0.0;
    double y = 0.0;
    for (std::uint64_t channel = 0; channel < hits.channels; ++channel) {
      const double count =
          static_cast<double>(hits.counts[event * hits.channels + channel]);
      total += count;
      x += count * channel_xy[2 * channel];
      y += count * channel_xy[2 * channel + 1];
    }
    if (total > 0.0) {
      x /= total;
      y /= total;
    }
    if (domain.shape == "disk") {
      const double radial = std::hypot(x, y);
      if (radial > domain.radius && radial > 0.0) {
        x *= domain.radius / radial;
        y *= domain.radius / radial;
      }
    } else if (domain.shape == "rectangle") {
      x = std::clamp(x, -domain.half_x, domain.half_x);
      y = std::clamp(y, -domain.half_y, domain.half_y);
    } else {
      x = std::clamp(x, -domain.half_x, domain.half_x);
      const double coordinate =
          (y - domain.line_y_start) / domain.line_pitch;
      const auto line = std::clamp<std::int64_t>(
          static_cast<std::int64_t>(std::llround(coordinate)), 0,
          static_cast<std::int64_t>(domain.line_count) - 1);
      y = domain.line_y_start + line * domain.line_pitch;
    }
    result.fitted_xy_mm[2 * event] = x;
    result.fitted_xy_mm[2 * event + 1] = y;
    if (domain.shape == "parallel_lines") {
      result.fitted_line_id[event] =
          static_cast<std::int32_t>(std::llround(
              (y - domain.line_y_start) / domain.line_pitch));
      result.fitted_line_x_mm[event] = x;
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    const auto device = option(argc, argv, "--device", "cuda");
    if (command == "grid") {
      if (argc < 3) throw std::runtime_error("operators.h5 is required");
      const auto precomputed = option(argc, argv, "--precomputed");
      const auto scene_path = option(argc, argv, "--scene");
      const auto output = option(argc, argv, "--output");
      if (precomputed.empty() || scene_path.empty() || output.empty())
        throw std::runtime_error(
            "--precomputed, --scene, and --output are required");
      auto operators = oos::load_operators_hdf5(argv[2]);
      auto response = oos::load_effective_response_hdf5(precomputed);
      auto scene = configured_scene(argc, argv, scene_path);
      const double source_z =
          std::stod(option(argc, argv, "--source-z-mm", "2.5"));
      const double source_thickness =
          std::stod(option(argc, argv, "--source-thickness-mm", "5"));
      if (source_thickness < 0.0)
        throw std::runtime_error(
            "--source-thickness-mm must be nonnegative");
      oos::ShapeFactorOptions shape_factor;
      shape_factor.backend = shape_factor_backend(
          option(argc, argv, "--source-backend", "auto"));
      shape_factor.maximum_subdivision_depth =
          static_cast<std::uint32_t>(std::stoul(
              option(argc, argv, "--source-subdivision-depth", "8")));
      shape_factor.relative_tolerance = std::stod(
          option(argc, argv, "--source-relative-tolerance", "1e-5"));
      shape_factor.structured_disk_mu_order =
          static_cast<std::uint32_t>(std::stoul(
              option(argc, argv, "--structured-disk-mu-order", "31")));
      shape_factor.structured_disk_phi_count =
          static_cast<std::uint32_t>(std::stoul(
              option(argc, argv, "--structured-disk-phi-count", "64")));
      const auto source_angular_mode =
          option(argc, argv, "--source-angular-mode", "shape_factor");
      const auto source_transverse_count = static_cast<std::uint32_t>(
          std::stoul(option(argc, argv, "--source-transverse-count", "32")));
      const double obstacle_half_width =
          std::stod(option(argc, argv, "--obstacle-half-width-mm", "0.0025"));
      const double obstacle_half_thickness =
          std::stod(option(argc, argv, "--obstacle-half-thickness-mm", "0.0005"));
      const double source_medium_z_max =
          std::stod(option(argc, argv, "--source-medium-z-max-mm", "17.5"));
      const auto source_mu_order = static_cast<std::uint32_t>(
          std::stoul(option(argc, argv, "--source-mu-order", "32")));
      const auto source_phi_count = static_cast<std::uint32_t>(
          std::stoul(option(argc, argv, "--source-phi-count", "128")));
      CandidateEvaluator evaluator(std::move(scene), std::move(operators),
                                   std::move(response), device, source_z,
                                   source_thickness, shape_factor,
                                   source_angular_mode,
                                   source_transverse_count, obstacle_half_width,
                                   obstacle_half_thickness, source_medium_z_max,
                                   source_mu_order,
                                   source_phi_count);
      const double spacing =
          std::stod(option(argc, argv, "--spacing-mm", "10"));
      const auto domain = search_domain_from_arguments(argc, argv);
      const auto batch_size = source_batch_size(
          option(argc, argv, "--batch-size", "auto"),
          evaluator.response());
      std::vector<double> xy;
      if (domain.shape == "disk")
        xy = oos::cartesian_disk_grid(domain.radius, spacing);
      else if (domain.shape == "rectangle")
        xy = oos::cartesian_rectangle_grid(domain.half_x, domain.half_y,
                                           spacing);
      else
        xy = oos::cartesian_parallel_line_grid(
            domain.half_x, spacing, domain.line_y_start, domain.line_pitch,
            domain.line_count);
      std::vector<double> efficiency(
          xy.size() / 2 * evaluator.response().channels);
      for (std::uint64_t start = 0; start < xy.size() / 2;
           start += batch_size) {
        const auto stop =
            std::min<std::uint64_t>(xy.size() / 2, start + batch_size);
        std::vector<double> batch(
            xy.begin() + 2 * start, xy.begin() + 2 * stop);
        const auto evaluated = evaluator.evaluate(batch);
        std::copy(evaluated.efficiency.begin(), evaluated.efficiency.end(),
                  efficiency.begin() +
                      start * evaluator.response().channels);
        std::cout << "grid " << stop << "/" << xy.size() / 2 << "\n";
      }
      auto grid = oos::make_response_grid(
          xy, efficiency, evaluator.response().channels,
          evaluator.response().channel_ids, domain.radius, spacing, 1.0e-15,
          domain.shape, domain.half_x, domain.half_y, domain.line_y_start,
          domain.line_pitch, domain.line_count);
      grid.effective_response_fingerprint_sha256 =
          oos::effective_response_fingerprint(evaluator.response());
      if (flag(argc, argv, "--verify-effective-content"))
        grid.effective_response_sha256 = oos::sha256_file(precomputed);
      grid.source_angular_mode = source_angular_mode;
      grid.source_backend = shape_factor_backend_name(shape_factor.backend);
      grid.source_z_mm = source_z;
      grid.source_thickness_mm = source_thickness;
      grid.source_transverse_count = source_transverse_count;
      grid.obstacle_half_width_mm = obstacle_half_width;
      grid.obstacle_half_thickness_mm = obstacle_half_thickness;
      grid.source_medium_z_max_mm = source_medium_z_max;
      grid.source_mu_order = source_mu_order;
      grid.source_phi_count = source_phi_count;
      grid.source_relative_tolerance = shape_factor.relative_tolerance;
      grid.source_maximum_subdivision_depth =
          shape_factor.maximum_subdivision_depth;
      grid.structured_disk_mu_order =
          shape_factor.structured_disk_mu_order;
      grid.structured_disk_phi_count =
          shape_factor.structured_disk_phi_count;
      grid.fingerprint_sha256 = oos::response_grid_fingerprint(grid);
      auto temporary = std::filesystem::path(output);
      temporary += ".tmp";
      oos::save_response_grid_hdf5(temporary, grid);
      std::filesystem::rename(temporary, output);
      return 0;
    }
    if (command == "fit") {
      const auto hits_path = option(argc, argv, "--hits");
      const auto output = option(argc, argv, "--output");
      if (hits_path.empty() || output.empty())
        throw std::runtime_error("--hits and --output are required");
      const auto hits = oos::load_hit_batch_hdf5(hits_path);
      const auto fit_mode = option(argc, argv, "--fit-mode", "fast");
      if (fit_mode != "accurate" && fit_mode != "fast")
        throw std::runtime_error("--fit-mode must be accurate or fast");
      oos::RegressionResult result;
      const auto grid_path = option(argc, argv, "--grid");
      std::unique_ptr<CandidateEvaluator> evaluator;
      std::unique_ptr<oos::ResponseGrid> response_grid;
      auto domain = search_domain_from_arguments(argc, argv);
      if (!grid_path.empty()) {
        response_grid = std::make_unique<oos::ResponseGrid>(
            oos::load_response_grid_hdf5(grid_path));
        const double coarse_spacing =
            std::stod(option(argc, argv, "--coarse-spacing-mm", "50"));
        const auto coarse_grid =
            coarse_response_grid(*response_grid, coarse_spacing);
        std::vector<double> scores;
        if (device == "cpu")
          scores = oos::score_response_grid_cpu(coarse_grid, hits);
        else {
#ifdef OOS_HAS_CUDA
          scores = oos::score_response_grid_cuda(coarse_grid, hits);
#else
          throw std::runtime_error("this build has no CUDA backend");
#endif
        }
        result = oos::fit_response_grid_scores(
            coarse_grid, hits, std::move(scores), flag(argc, argv, "--adjoint"));
        domain = search_domain_from_grid(*response_grid);
      }
      if (fit_mode == "fast") {
        if (!response_grid)
          throw std::runtime_error("fast fit mode requires --grid");
        refine_events_fast(*response_grid, hits, domain,
                           refinement_levels(argc, argv, true), result);
      } else {
        const auto operator_path = option(argc, argv, "--operators");
        const auto precomputed = option(argc, argv, "--precomputed");
        const auto scene_path = option(argc, argv, "--scene");
        if (operator_path.empty() || precomputed.empty() ||
            scene_path.empty())
          throw std::runtime_error(
              "continuous regression requires --operators, --precomputed, "
              "and --scene");
        auto operators = oos::load_operators_hdf5(operator_path);
        auto response = oos::load_effective_response_hdf5(precomputed);
        auto scene = configured_scene(argc, argv, scene_path);
        const double source_z =
            std::stod(option(argc, argv, "--source-z-mm", "2.5"));
        const double source_thickness =
            std::stod(option(argc, argv, "--source-thickness-mm", "5"));
        if (source_thickness < 0.0)
          throw std::runtime_error(
              "--source-thickness-mm must be nonnegative");
        oos::ShapeFactorOptions shape_factor;
        shape_factor.backend = shape_factor_backend(
            option(argc, argv, "--source-backend", "auto"));
        shape_factor.maximum_subdivision_depth =
            static_cast<std::uint32_t>(std::stoul(
                option(argc, argv, "--source-subdivision-depth", "8")));
        shape_factor.relative_tolerance = std::stod(
            option(argc, argv, "--source-relative-tolerance", "1e-5"));
        shape_factor.structured_disk_mu_order =
            static_cast<std::uint32_t>(std::stoul(
                option(argc, argv, "--structured-disk-mu-order", "31")));
        shape_factor.structured_disk_phi_count =
            static_cast<std::uint32_t>(std::stoul(
                option(argc, argv, "--structured-disk-phi-count", "64")));
        const auto source_angular_mode =
            option(argc, argv, "--source-angular-mode", "shape_factor");
        const auto source_transverse_count = static_cast<std::uint32_t>(
            std::stoul(option(argc, argv, "--source-transverse-count", "32")));
        const double obstacle_half_width =
            std::stod(option(argc, argv, "--obstacle-half-width-mm", "0.0025"));
        const double obstacle_half_thickness =
            std::stod(
                option(argc, argv, "--obstacle-half-thickness-mm", "0.0005"));
        const double source_medium_z_max =
            std::stod(option(argc, argv, "--source-medium-z-max-mm", "17.5"));
        const auto source_mu_order = static_cast<std::uint32_t>(
            std::stoul(option(argc, argv, "--source-mu-order", "32")));
        const auto source_phi_count = static_cast<std::uint32_t>(
            std::stoul(option(argc, argv, "--source-phi-count", "128")));
        const auto batch_size = source_batch_size(
            option(argc, argv, "--batch-size", "auto"), response);
        if (grid_path.empty())
          result = centroid_initial(scene, response, hits, domain);
        evaluator = std::make_unique<CandidateEvaluator>(
            std::move(scene), std::move(operators), std::move(response),
            device, source_z, source_thickness, shape_factor,
            source_angular_mode, source_transverse_count, obstacle_half_width,
            obstacle_half_thickness, source_medium_z_max, source_mu_order,
            source_phi_count);
        const bool default_refinement =
            option(argc, argv, "--refine-spacings-mm").empty() &&
            option(argc, argv, "--refine-half-widths-mm").empty();
        refine_events_accurate(
            *evaluator, hits, domain,
            refinement_levels(argc, argv, false), default_refinement,
            batch_size, result);
      }
      auto temporary = std::filesystem::path(output);
      temporary += ".tmp";
      oos::save_regression_hdf5(temporary, result, hits.channel_ids);
      std::filesystem::rename(temporary, output);
      std::cout << "fit " << hits.count << " events\n";
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& exception) {
    std::cerr << "oos-regress: " << exception.what() << '\n';
    return 1;
  }
}
