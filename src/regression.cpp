#include "oos/regression.hpp"

#include "oos/hash.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace oos {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::vector<std::pair<double, double>> gauss_legendre_unit_interval(
    std::uint32_t count) {
  if (count == 0)
    throw std::runtime_error("Gauss-Legendre order must be positive");
  std::vector<std::pair<double, double>> result(count);
  const std::uint32_t half = (count + 1u) / 2u;
  for (std::uint32_t i = 0; i < half; ++i) {
    double root =
        std::cos(pi * (static_cast<double>(i) + 0.75) /
                 (static_cast<double>(count) + 0.5));
    double derivative = 0.0;
    for (std::uint32_t iteration = 0; iteration < 64; ++iteration) {
      double previous = 1.0;
      double value = root;
      for (std::uint32_t order = 2; order <= count; ++order) {
        const double next =
            ((2.0 * order - 1.0) * root * value -
             (order - 1.0) * previous) /
            order;
        previous = value;
        value = next;
      }
      derivative =
          count * (root * value - previous) / (root * root - 1.0);
      const double updated = root - value / derivative;
      if (std::abs(updated - root) <= 4.0e-16) {
        root = updated;
        break;
      }
      root = updated;
    }
    const double weight =
        1.0 / ((1.0 - root * root) * derivative * derivative);
    result[i] = {(1.0 - root) * 0.5, weight};
    result[count - 1u - i] = {(1.0 + root) * 0.5, weight};
  }
  return result;
}

Vec3 reference_axis(const Vec3& direction) {
  if (std::abs(direction.z) < 0.9) {
    const double norm = std::hypot(direction.x, direction.y);
    return {-direction.y / norm, direction.x / norm, 0.0};
  }
  const double norm = std::hypot(direction.x, direction.z);
  return {direction.z / norm, 0.0, -direction.x / norm};
}

struct GridKey {
  std::int64_t x{};
  std::int64_t y{};

  bool operator==(const GridKey& other) const {
    return x == other.x && y == other.y;
  }
};

struct GridKeyHash {
  std::size_t operator()(const GridKey& key) const {
    const auto first = std::hash<std::int64_t>{}(key.x);
    const auto second = std::hash<std::int64_t>{}(key.y);
    return first ^ (second + 0x9e3779b97f4a7c15ull + (first << 6u) +
                    (first >> 2u));
  }
};

struct VertexBlend {
  std::array<std::uint64_t, 3> index{};
  std::array<double, 3> weight{};
  std::uint32_t count{};
};

bool barycentric_triangle(
    double x, double y, const std::array<std::array<double, 2>, 3>& vertex,
    std::array<double, 3>& weight) {
  const double denominator =
      (vertex[1][1] - vertex[2][1]) *
          (vertex[0][0] - vertex[2][0]) +
      (vertex[2][0] - vertex[1][0]) *
          (vertex[0][1] - vertex[2][1]);
  if (std::abs(denominator) <= 1.0e-18) return false;
  weight[0] =
      ((vertex[1][1] - vertex[2][1]) * (x - vertex[2][0]) +
       (vertex[2][0] - vertex[1][0]) * (y - vertex[2][1])) /
      denominator;
  weight[1] =
      ((vertex[2][1] - vertex[0][1]) * (x - vertex[2][0]) +
       (vertex[0][0] - vertex[2][0]) * (y - vertex[2][1])) /
      denominator;
  weight[2] = 1.0 - weight[0] - weight[1];
  constexpr double tolerance = 1.0e-10;
  return std::all_of(weight.begin(), weight.end(), [](double value) {
    return value >= -tolerance && value <= 1.0 + tolerance;
  });
}

}  // namespace

LocalQuadraticPeak fit_local_quadratic_peak(
    const std::vector<double>& xy_mm, const std::vector<double>& values,
    std::uint64_t best_index, double spacing_mm, bool one_dimensional) {
  if (values.empty() || xy_mm.size() != values.size() * 2 ||
      best_index >= values.size() || !(spacing_mm > 0.0) ||
      !std::isfinite(spacing_mm))
    throw std::runtime_error("invalid sampled likelihood surface");
  LocalQuadraticPeak result{xy_mm[2 * best_index],
                            xy_mm[2 * best_index + 1], values[best_index],
                            false};
  const double x0 = result.x_mm;
  const double y0 = result.y_mm;
  const double f0 = result.log_likelihood;
  const double coordinate_tolerance =
      std::max(1.0e-10, spacing_mm * 1.0e-8);
  const auto sampled = [&](double x, double y, double& value) {
    for (std::uint64_t index = 0; index < values.size(); ++index)
      if (std::abs(xy_mm[2 * index] - x) <= coordinate_tolerance &&
          std::abs(xy_mm[2 * index + 1] - y) <= coordinate_tolerance) {
        value = values[index];
        return std::isfinite(value);
      }
    return false;
  };
  double fx_minus{}, fx_plus{};
  if (!sampled(x0 - spacing_mm, y0, fx_minus) ||
      !sampled(x0 + spacing_mm, y0, fx_plus))
    return result;
  const double inverse_two_h = 0.5 / spacing_mm;
  const double inverse_h2 = 1.0 / (spacing_mm * spacing_mm);
  const double gradient_x = (fx_plus - fx_minus) * inverse_two_h;
  const double hessian_xx = (fx_plus - 2.0 * f0 + fx_minus) * inverse_h2;
  const double curvature_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(f0)) * inverse_h2;
  if (one_dimensional) {
    if (!(hessian_xx < -curvature_tolerance)) return result;
    const double dx = -gradient_x / hessian_xx;
    if (!std::isfinite(dx) || std::abs(dx) > spacing_mm) return result;
    const double fitted =
        f0 + gradient_x * dx + 0.5 * hessian_xx * dx * dx;
    if (!std::isfinite(fitted) || fitted < f0) return result;
    return {x0 + dx, y0, fitted, true};
  }

  double fy_minus{}, fy_plus{}, fmm{}, fmp{}, fpm{}, fpp{};
  if (!sampled(x0, y0 - spacing_mm, fy_minus) ||
      !sampled(x0, y0 + spacing_mm, fy_plus) ||
      !sampled(x0 - spacing_mm, y0 - spacing_mm, fmm) ||
      !sampled(x0 - spacing_mm, y0 + spacing_mm, fmp) ||
      !sampled(x0 + spacing_mm, y0 - spacing_mm, fpm) ||
      !sampled(x0 + spacing_mm, y0 + spacing_mm, fpp))
    return result;
  const double gradient_y = (fy_plus - fy_minus) * inverse_two_h;
  const double hessian_yy = (fy_plus - 2.0 * f0 + fy_minus) * inverse_h2;
  const double hessian_xy =
      (fpp - fpm - fmp + fmm) * 0.25 * inverse_h2;
  const double determinant =
      hessian_xx * hessian_yy - hessian_xy * hessian_xy;
  if (!(hessian_xx < -curvature_tolerance) ||
      !(hessian_yy < -curvature_tolerance) ||
      !(determinant > curvature_tolerance * curvature_tolerance))
    return result;
  const double dx =
      (hessian_xy * gradient_y - hessian_yy * gradient_x) / determinant;
  const double dy =
      (hessian_xy * gradient_x - hessian_xx * gradient_y) / determinant;
  if (!std::isfinite(dx) || !std::isfinite(dy) ||
      std::abs(dx) > spacing_mm || std::abs(dy) > spacing_mm)
    return result;
  const double fitted =
      f0 + gradient_x * dx + gradient_y * dy +
      0.5 * (hessian_xx * dx * dx +
             2.0 * hessian_xy * dx * dy + hessian_yy * dy * dy);
  if (!std::isfinite(fitted) || fitted < f0) return result;
  return {x0 + dx, y0 + dy, fitted, true};
}

struct ResponseGridInterpolator::Impl {
  explicit Impl(const ResponseGrid& source) : grid(source) {
    grid.validate();
    absolute_efficiency.resize(grid.points * grid.channels);
    lattice.reserve(static_cast<std::size_t>(grid.points * 1.3));
    for (std::uint64_t point = 0; point < grid.points; ++point) {
      const double scaled_x = grid.xy_mm[2 * point] / grid.spacing_mm;
      double scaled_y = grid.xy_mm[2 * point + 1] / grid.spacing_mm;
      if (grid.domain_shape == "parallel_lines")
        scaled_y = (grid.xy_mm[2 * point + 1] - grid.line_y_start_mm) /
                   grid.line_pitch_mm;
      const auto ix = static_cast<std::int64_t>(std::llround(scaled_x));
      const auto iy = static_cast<std::int64_t>(std::llround(scaled_y));
      if (std::abs(scaled_x - ix) > 1.0e-8 ||
          std::abs(scaled_y - iy) > 1.0e-8 ||
          !lattice.emplace(GridKey{ix, iy}, point).second)
        throw std::runtime_error(
            "response grid is not a unique regular lattice");
      const auto offset = point * grid.channels;
      for (std::uint64_t channel = 0; channel < grid.channels; ++channel)
        absolute_efficiency[offset + channel] = static_cast<float>(
            grid.top_efficiency[point] *
            std::exp(static_cast<double>(
                grid.conditional_log_probability[offset + channel])));
    }
  }

  bool blend(double x_mm, double y_mm, VertexBlend& result) const {
    if (!std::isfinite(x_mm) || !std::isfinite(y_mm)) return false;
    const double gx = x_mm / grid.spacing_mm;
    double gy = y_mm / grid.spacing_mm;
    if (grid.domain_shape == "parallel_lines") {
      gy = (y_mm - grid.line_y_start_mm) / grid.line_pitch_mm;
      const auto line = static_cast<std::int64_t>(std::llround(gy));
      if (std::abs(gy - line) > 1.0e-8) return false;
      const auto rounded_x = static_cast<std::int64_t>(std::llround(gx));
      const auto exact = lattice.find({rounded_x, line});
      if (std::abs(gx - rounded_x) <= 1.0e-10 && exact != lattice.end()) {
        result.index[0] = exact->second;
        result.weight[0] = 1.0;
        result.count = 1;
        return true;
      }
      const auto left_x = static_cast<std::int64_t>(std::floor(gx));
      const auto left = lattice.find({left_x, line});
      const auto right = lattice.find({left_x + 1, line});
      if (left == lattice.end() || right == lattice.end()) return false;
      const double fraction = gx - left_x;
      result.index = {left->second, right->second, 0};
      result.weight = {1.0 - fraction, fraction, 0.0};
      result.count = 2;
      return true;
    }

    const auto rounded_x = static_cast<std::int64_t>(std::llround(gx));
    const auto rounded_y = static_cast<std::int64_t>(std::llround(gy));
    const auto exact = lattice.find({rounded_x, rounded_y});
    if (std::abs(gx - rounded_x) <= 1.0e-10 &&
        std::abs(gy - rounded_y) <= 1.0e-10 && exact != lattice.end()) {
      result.index[0] = exact->second;
      result.weight[0] = 1.0;
      result.count = 1;
      return true;
    }
    const auto ix = static_cast<std::int64_t>(std::floor(gx));
    const auto iy = static_cast<std::int64_t>(std::floor(gy));
    const std::array<GridKey, 4> keys{{{ix, iy},
                                       {ix + 1, iy},
                                       {ix, iy + 1},
                                       {ix + 1, iy + 1}}};
    std::array<std::uint64_t, 4> indices{};
    std::array<bool, 4> present{};
    for (std::uint32_t corner = 0; corner < 4; ++corner) {
      const auto found = lattice.find(keys[corner]);
      present[corner] = found != lattice.end();
      if (present[corner]) indices[corner] = found->second;
    }
    // Deterministic lower-left to upper-right diagonal, followed by the two
    // alternate boundary triangles needed by clipped disk cells.
    constexpr std::array<std::array<std::uint32_t, 3>, 4> triangles{{
        {{0, 1, 3}}, {{0, 3, 2}}, {{0, 1, 2}}, {{1, 3, 2}}}};
    for (const auto& triangle : triangles) {
      if (!present[triangle[0]] || !present[triangle[1]] ||
          !present[triangle[2]])
        continue;
      std::array<std::array<double, 2>, 3> vertex{};
      for (std::uint32_t item = 0; item < 3; ++item)
        vertex[item] = {static_cast<double>(keys[triangle[item]].x),
                        static_cast<double>(keys[triangle[item]].y)};
      std::array<double, 3> weight{};
      if (!barycentric_triangle(gx, gy, vertex, weight)) continue;
      for (std::uint32_t item = 0; item < 3; ++item)
        result.index[item] = indices[triangle[item]];
      result.weight = weight;
      result.count = 3;
      return true;
    }
    return false;
  }

  double score(double x_mm, double y_mm, const std::uint64_t* counts,
               std::uint64_t emitted) const {
    VertexBlend vertices;
    if (!blend(x_mm, y_mm, vertices))
      return -std::numeric_limits<double>::infinity();
    if (vertices.count == 1) {
      const auto point = vertices.index[0];
      double value = 0.0;
      std::uint64_t top_hits = 0;
      for (std::uint64_t channel = 0; channel < grid.channels; ++channel) {
        top_hits += counts[channel];
        value += static_cast<double>(counts[channel]) *
                 static_cast<double>(grid.conditional_log_probability[
                     point * grid.channels + channel]);
      }
      if (emitted != 0) {
        if (emitted < top_hits)
          throw std::runtime_error(
              "emitted count is smaller than observed top hits");
        const double total = grid.top_efficiency[point];
        value += static_cast<double>(top_hits) * std::log(total) +
                 static_cast<double>(emitted - top_hits) *
                     std::log1p(-total);
      }
      return value;
    }
    double total_efficiency = 0.0;
    for (std::uint32_t item = 0; item < vertices.count; ++item)
      total_efficiency += vertices.weight[item] *
                          grid.top_efficiency[vertices.index[item]];
    if (!(total_efficiency > 0.0) || !(total_efficiency < 1.0))
      return -std::numeric_limits<double>::infinity();
    constexpr double probability_floor = 1.0e-15;
    double value = 0.0;
    std::uint64_t top_hits = 0;
    for (std::uint64_t channel = 0; channel < grid.channels; ++channel) {
      double efficiency = 0.0;
      for (std::uint32_t item = 0; item < vertices.count; ++item)
        efficiency +=
            vertices.weight[item] *
            absolute_efficiency[vertices.index[item] * grid.channels +
                                channel];
      top_hits += counts[channel];
      const double probability =
          emitted == 0 ? efficiency / total_efficiency : efficiency;
      value += static_cast<double>(counts[channel]) *
               std::log(std::max(probability, probability_floor));
    }
    if (emitted != 0) {
      if (emitted < top_hits)
        throw std::runtime_error(
            "emitted count is smaller than observed top hits");
      value += static_cast<double>(emitted - top_hits) *
               std::log(std::max(1.0 - total_efficiency,
                                 probability_floor));
    }
    return value;
  }

  const ResponseGrid& grid;
  std::vector<float> absolute_efficiency;
  std::unordered_map<GridKey, std::uint64_t, GridKeyHash> lattice;
};

ResponseGridInterpolator::ResponseGridInterpolator(const ResponseGrid& grid)
    : impl_(std::make_unique<Impl>(grid)) {}

ResponseGridInterpolator::~ResponseGridInterpolator() = default;
ResponseGridInterpolator::ResponseGridInterpolator(
    ResponseGridInterpolator&&) noexcept = default;
ResponseGridInterpolator& ResponseGridInterpolator::operator=(
    ResponseGridInterpolator&&) noexcept = default;

double ResponseGridInterpolator::score(double x_mm, double y_mm,
                                       const std::uint64_t* counts,
                                       std::uint64_t emitted) const {
  return impl_->score(x_mm, y_mm, counts, emitted);
}

std::string response_grid_fingerprint(const ResponseGrid& grid) {
  std::ostringstream material;
  material << "oos.response-grid.semantic.v2\n"
           << grid.effective_response_fingerprint_sha256 << '\n'
           << grid.domain_shape << '\n'
           << std::setprecision(17) << grid.radius_mm << '\n'
           << grid.half_x_mm << '\n'
           << grid.half_y_mm << '\n'
           << grid.line_y_start_mm << '\n'
           << grid.line_pitch_mm << '\n'
           << grid.line_count << '\n'
           << grid.spacing_mm << '\n'
           << grid.points << '\n'
           << grid.channels << '\n'
           << grid.source_angular_mode << '\n'
           << grid.source_backend << '\n'
           << grid.source_z_mm << '\n'
           << grid.source_thickness_mm << '\n'
           << grid.source_transverse_count << '\n'
           << grid.obstacle_half_width_mm << '\n'
           << grid.obstacle_half_thickness_mm << '\n'
           << grid.source_medium_z_max_mm << '\n'
           << grid.source_mu_order << '\n'
           << grid.source_phi_count << '\n'
           << grid.source_relative_tolerance << '\n'
           << grid.source_maximum_subdivision_depth << '\n'
           << grid.structured_disk_mu_order << '\n'
           << grid.structured_disk_phi_count;
  for (const auto channel : grid.channel_ids) material << '\n' << channel;
  return sha256_string(material.str());
}

void ResponseGrid::validate() const {
  const bool valid_domain =
      domain_shape == "disk"
          ? radius_mm > 0.0
          : domain_shape == "rectangle"
                ? half_x_mm > 0.0 && half_y_mm > 0.0
                : domain_shape == "parallel_lines" && half_x_mm > 0.0 &&
                      std::isfinite(line_y_start_mm) &&
                      line_pitch_mm > 0.0 && line_count > 0;
  const bool valid_backend =
      source_backend == "auto" || source_backend == "generic_bvh" ||
      source_backend == "structured_analytic";
  if (!valid_domain || !(spacing_mm > 0.0) || points == 0 ||
      channels == 0 || xy_mm.size() != points * 2 ||
      conditional_log_probability.size() != points * channels ||
      top_efficiency.size() != points || channel_ids.size() != channels ||
      !valid_backend || !(source_relative_tolerance > 0.0) ||
      structured_disk_mu_order < 2 || structured_disk_phi_count < 2)
    throw std::runtime_error("response grid dimensions are invalid");
  if (!std::all_of(
          conditional_log_probability.begin(),
          conditional_log_probability.end(),
          [](float value) { return std::isfinite(value); }) ||
      !std::all_of(top_efficiency.begin(), top_efficiency.end(),
                   [](double value) {
                     return std::isfinite(value) && value > 0.0 &&
                            value < 1.0;
                   }))
    throw std::runtime_error("response grid contains invalid values");
}

std::vector<double> cartesian_disk_grid(double radius_mm,
                                        double spacing_mm) {
  if (!std::isfinite(radius_mm) || !std::isfinite(spacing_mm) ||
      radius_mm <= 0.0 || spacing_mm <= 0.0)
    throw std::runtime_error("grid radius and spacing must be positive");
  const auto extent =
      static_cast<std::int64_t>(std::floor(radius_mm / spacing_mm));
  std::vector<double> result;
  result.reserve(static_cast<std::size_t>(
      3.2 * radius_mm * radius_mm / (spacing_mm * spacing_mm)));
  for (std::int64_t iy = -extent; iy <= extent; ++iy) {
    const double y = spacing_mm * iy;
    for (std::int64_t ix = -extent; ix <= extent; ++ix) {
      const double x = spacing_mm * ix;
      if (x * x + y * y <= radius_mm * radius_mm + 1.0e-9) {
        result.push_back(x);
        result.push_back(y);
      }
    }
  }
  return result;
}

std::vector<double> cartesian_rectangle_grid(double half_x_mm,
                                             double half_y_mm,
                                             double spacing_mm) {
  if (!std::isfinite(half_x_mm) || !std::isfinite(half_y_mm) ||
      !std::isfinite(spacing_mm) || half_x_mm <= 0.0 || half_y_mm <= 0.0 ||
      spacing_mm <= 0.0)
    throw std::runtime_error(
        "grid half-widths and spacing must be positive");
  const auto x_extent =
      static_cast<std::int64_t>(std::floor(half_x_mm / spacing_mm));
  const auto y_extent =
      static_cast<std::int64_t>(std::floor(half_y_mm / spacing_mm));
  std::vector<double> result;
  result.reserve(static_cast<std::size_t>(
      (2 * x_extent + 1) * (2 * y_extent + 1) * 2));
  for (std::int64_t iy = -y_extent; iy <= y_extent; ++iy) {
    const double y = spacing_mm * iy;
    for (std::int64_t ix = -x_extent; ix <= x_extent; ++ix) {
      result.push_back(spacing_mm * ix);
      result.push_back(y);
    }
  }
  return result;
}

std::vector<double> cartesian_parallel_line_grid(double half_x_mm, double spacing_mm,
                                        double line_y_start_mm,
                                        double line_pitch_mm,
                                        std::uint64_t line_count) {
  if (!std::isfinite(half_x_mm) || !std::isfinite(spacing_mm) ||
      !std::isfinite(line_y_start_mm) ||
      !std::isfinite(line_pitch_mm) || half_x_mm <= 0.0 ||
      spacing_mm <= 0.0 || line_pitch_mm <= 0.0 || line_count == 0)
    throw std::runtime_error("line-grid parameters must be finite and positive");
  const auto x_extent =
      static_cast<std::int64_t>(std::floor(half_x_mm / spacing_mm));
  std::vector<double> result;
  result.reserve(static_cast<std::size_t>(
      line_count * static_cast<std::uint64_t>(2 * x_extent + 1) * 2));
  for (std::uint64_t line = 0; line < line_count; ++line) {
    const double y = line_y_start_mm + line_pitch_mm * line;
    for (std::int64_t ix = -x_extent; ix <= x_extent; ++ix) {
      result.push_back(spacing_mm * ix);
      result.push_back(y);
    }
  }
  return result;
}

std::vector<SourceQuadrature> make_xy_shape_factor_sources(
    const std::vector<double>& xy_mm, std::int32_t domain,
    double z_min_mm, double z_max_mm, ShapeFactorOptions options) {
  if (xy_mm.size() % 2 != 0 || z_max_mm < z_min_mm)
    throw std::runtime_error("invalid XY source coordinates or z interval");
  // Four-point Gauss-Legendre rule mapped to [z_min,z_max].
  constexpr std::array<double, 4> nodes{
      -0.8611363115940525752, -0.3399810435848562648,
      0.3399810435848562648, 0.8611363115940525752};
  constexpr std::array<double, 4> weights{
      0.3478548451374538574, 0.6521451548625461426,
      0.6521451548625461426, 0.3478548451374538574};
  const double midpoint = 0.5 * (z_min_mm + z_max_mm);
  const double half = 0.5 * (z_max_mm - z_min_mm);
  std::vector<SourceQuadrature> result(xy_mm.size() / 2);
  for (std::size_t index = 0; index < result.size(); ++index) {
    auto& source = result[index];
    source.id = "xy-" + std::to_string(index);
    source.integration = SourceIntegration::isotropic_surface_shape_factor;
    source.shape_factor = options;
    if (z_max_mm == z_min_mm) {
      source.points.push_back(
          {{xy_mm[2 * index], xy_mm[2 * index + 1], z_min_mm},
           {1.0, 0.0, 0.0, 0.0},
           domain});
      continue;
    }
    source.points.reserve(nodes.size());
    for (std::size_t z = 0; z < nodes.size(); ++z)
      source.points.push_back(
          {{xy_mm[2 * index], xy_mm[2 * index + 1],
            midpoint + half * nodes[z]},
           {0.5 * weights[z], 0.0, 0.0, 0.0},
           domain});
  }
  return result;
}

std::vector<SourceQuadrature> make_xy_isotropic_product_sources(
    const std::vector<double>& xy_mm, std::int32_t domain, double z_mm,
    std::uint32_t mu_order, std::uint32_t phi_count) {
  if (xy_mm.size() % 2 != 0 || !std::isfinite(z_mm) || mu_order == 0 ||
      phi_count == 0)
    throw std::runtime_error(
        "invalid XY source coordinates or isotropic-product order");
  const auto mu_nodes = gauss_legendre_unit_interval(mu_order);
  std::vector<SourceQuadrature> result(xy_mm.size() / 2);
  for (std::size_t index = 0; index < result.size(); ++index) {
    auto& source = result[index];
    source.id = "xy-" + std::to_string(index);
    source.integration = SourceIntegration::discrete_rays;
    source.rays.reserve(static_cast<std::size_t>(mu_order) * phi_count);
    for (const auto& [u, half_weight] : mu_nodes) {
      const double mu = 2.0 * u - 1.0;
      const double transverse =
          std::sqrt(std::max(0.0, 1.0 - mu * mu));
      for (std::uint32_t phi_index = 0; phi_index < phi_count; ++phi_index) {
        const double phi =
            2.0 * pi * (static_cast<double>(phi_index) + 0.5) / phi_count;
        const Vec3 direction{transverse * std::cos(phi),
                             transverse * std::sin(phi), mu};
        source.rays.push_back(
            {{{xy_mm[2 * index], xy_mm[2 * index + 1], z_mm}, direction},
             {half_weight / phi_count, 0.0, 0.0, 0.0},
             reference_axis(direction),
             domain});
      }
    }
  }
  return result;
}

std::vector<SourceQuadrature>
make_xy_rectangular_line_neighborhood_isotropic_product_sources(
    const std::vector<double>& xy_mm, std::int32_t domain,
    double line_center_z_mm, double obstacle_half_width_mm,
    double obstacle_half_thickness_mm, double maximum_distance_mm,
    double medium_z_max_mm, std::uint32_t transverse_count,
    std::uint32_t mu_order, std::uint32_t phi_count) {
  if (xy_mm.size() % 2 != 0 || !std::isfinite(line_center_z_mm) ||
      transverse_count == 0 || mu_order == 0 || phi_count == 0)
    throw std::runtime_error(
        "invalid line source coordinates or quadrature order");
  const auto mu_nodes = gauss_legendre_unit_interval(mu_order);
  std::vector<SourceQuadrature> result(xy_mm.size() / 2);
  for (std::size_t index = 0; index < result.size(); ++index) {
    auto& source = result[index];
    source.id = "line-" + std::to_string(index);
    source.integration = SourceIntegration::discrete_rays;
    const auto positions = uniform_rectangular_line_neighborhood_samples(
        {xy_mm[2 * index], xy_mm[2 * index + 1], line_center_z_mm},
        obstacle_half_width_mm, obstacle_half_thickness_mm, maximum_distance_mm,
        medium_z_max_mm, transverse_count);
    source.rays.reserve(static_cast<std::size_t>(transverse_count) *
                        mu_order * phi_count);
    for (const auto& position : positions) {
      for (const auto& [mu_u, half_weight] : mu_nodes) {
        const double mu = 2.0 * mu_u - 1.0;
        const double transverse =
            std::sqrt(std::max(0.0, 1.0 - mu * mu));
        for (std::uint32_t phi_index = 0; phi_index < phi_count;
             ++phi_index) {
          const double phi =
              2.0 * pi * (static_cast<double>(phi_index) + 0.5) /
              phi_count;
          const Vec3 direction{transverse * std::cos(phi),
                               transverse * std::sin(phi), mu};
          source.rays.push_back(
              {{position, direction},
               {half_weight /
                    (static_cast<double>(transverse_count) * phi_count),
                0.0, 0.0, 0.0},
               reference_axis(direction),
               domain});
        }
      }
    }
  }
  return result;
}

ResponseGrid make_response_grid(const std::vector<double>& xy_mm,
                                const std::vector<double>& efficiency,
                                std::uint64_t channels,
                                std::vector<std::int32_t> channel_ids,
                                double radius_mm, double spacing_mm,
                                double probability_floor,
                                std::string domain_shape,
                                double half_x_mm,
                                double half_y_mm,
                                double line_y_start_mm,
                                double line_pitch_mm,
                                std::uint64_t line_count) {
  if (xy_mm.size() % 2 != 0 || channels == 0 ||
      efficiency.size() != xy_mm.size() / 2 * channels ||
      channel_ids.size() != channels || !(probability_floor > 0.0) ||
      !(probability_floor < 1.0))
    throw std::runtime_error("cannot construct response grid");
  ResponseGrid grid;
  grid.domain_shape = std::move(domain_shape);
  grid.radius_mm = radius_mm;
  grid.half_x_mm = half_x_mm;
  grid.half_y_mm = half_y_mm;
  grid.line_y_start_mm = line_y_start_mm;
  grid.line_pitch_mm = line_pitch_mm;
  grid.line_count = line_count;
  grid.spacing_mm = spacing_mm;
  grid.points = xy_mm.size() / 2;
  grid.channels = channels;
  grid.xy_mm = xy_mm;
  grid.channel_ids = std::move(channel_ids);
  grid.top_efficiency.resize(grid.points);
  grid.conditional_log_probability.resize(grid.points * channels);
#pragma omp parallel for schedule(static)
  for (std::int64_t point = 0;
       point < static_cast<std::int64_t>(grid.points); ++point) {
    const auto offset = static_cast<std::uint64_t>(point) * channels;
    const double total =
        std::accumulate(efficiency.begin() + offset,
                        efficiency.begin() + offset + channels, 0.0);
    if (!(total > 0.0) || !(total < 1.0))
      continue;
    grid.top_efficiency[point] = total;
    for (std::uint64_t channel = 0; channel < channels; ++channel) {
      const double probability =
          std::max(efficiency[offset + channel] / total, probability_floor);
      grid.conditional_log_probability[offset + channel] =
          static_cast<float>(std::log(probability));
    }
  }
  grid.validate();
  return grid;
}

std::vector<double> score_response_grid_cpu(const ResponseGrid& grid,
                                            const HitBatch& hits) {
  grid.validate();
  if (hits.channels != grid.channels ||
      hits.channel_ids != grid.channel_ids)
    throw std::runtime_error("hit channels do not match response grid");
  if (!hits.emitted.empty() && hits.emitted.size() != hits.count)
    throw std::runtime_error("emitted counts do not match hit events");
  if (!hits.emitted.empty())
    for (std::uint64_t event = 0; event < hits.count; ++event) {
      const auto begin = hits.counts.begin() + event * hits.channels;
      const auto top_hits =
          std::accumulate(begin, begin + hits.channels, std::uint64_t{0});
      if (hits.emitted[event] < top_hits)
        throw std::runtime_error(
            "emitted count is smaller than observed top hits");
    }
  std::vector<double> result(hits.count * grid.points, 0.0);
#pragma omp parallel for schedule(static)
  for (std::int64_t event = 0;
       event < static_cast<std::int64_t>(hits.count); ++event) {
    const auto count_offset = static_cast<std::uint64_t>(event) * grid.channels;
    const auto top_hits = std::accumulate(
        hits.counts.begin() + count_offset,
        hits.counts.begin() + count_offset + grid.channels, std::uint64_t{0});
    for (std::uint64_t point = 0; point < grid.points; ++point) {
      double value = 0.0;
      for (std::uint64_t channel = 0; channel < grid.channels; ++channel)
        value += static_cast<double>(
                     grid.conditional_log_probability[
                         point * grid.channels + channel]) *
                 static_cast<double>(
                     hits.counts[event * grid.channels + channel]);
      if (!hits.emitted.empty()) {
        const double total = grid.top_efficiency[point];
        value += static_cast<double>(top_hits) * std::log(total) +
                 static_cast<double>(hits.emitted[event] - top_hits) *
                     std::log1p(-total);
      }
      result[event * grid.points + point] = value;
    }
  }
  return result;
}

RegressionResult fit_response_grid(const ResponseGrid& grid,
                                   const HitBatch& hits,
                                   bool retain_full_plane) {
  return fit_response_grid_scores(grid, hits,
                                  score_response_grid_cpu(grid, hits),
                                  retain_full_plane);
}

RegressionResult fit_response_grid_scores(
    const ResponseGrid& grid, const HitBatch& hits,
    std::vector<double> likelihood, bool retain_full_plane) {
  if (likelihood.size() != hits.count * grid.points)
    throw std::runtime_error("likelihood map dimensions are invalid");
  RegressionResult result;
  result.events = hits.count;
  result.fit_mode = "grid_discrete";
  result.likelihood = hits.emitted.empty() ? "conditional_multinomial"
                                           : "absolute_multinomial";
  result.fitted_xy_mm.resize(hits.count * 2);
  if (grid.domain_shape == "parallel_lines") {
    result.fitted_line_id.resize(hits.count);
    result.fitted_line_x_mm.resize(hits.count);
  }
  result.log_likelihood.resize(hits.count);
  if (!hits.truth_xy_mm.empty()) result.error_mm.resize(hits.count);
  for (std::uint64_t event = 0; event < hits.count; ++event) {
    const auto begin = likelihood.begin() + event * grid.points;
    const auto best =
        static_cast<std::uint64_t>(std::max_element(begin, begin + grid.points) -
                                   begin);
    result.fitted_xy_mm[2 * event] = grid.xy_mm[2 * best];
    result.fitted_xy_mm[2 * event + 1] = grid.xy_mm[2 * best + 1];
    if (grid.domain_shape == "parallel_lines") {
      result.fitted_line_id[event] = static_cast<std::int32_t>(std::llround(
          (result.fitted_xy_mm[2 * event + 1] - grid.line_y_start_mm) /
          grid.line_pitch_mm));
      result.fitted_line_x_mm[event] =
          result.fitted_xy_mm[2 * event];
    }
    result.log_likelihood[event] = likelihood[event * grid.points + best];
    if (!hits.truth_xy_mm.empty()) {
      const double dx =
          result.fitted_xy_mm[2 * event] - hits.truth_xy_mm[2 * event];
      const double dy = result.fitted_xy_mm[2 * event + 1] -
                        hits.truth_xy_mm[2 * event + 1];
      result.error_mm[event] = std::hypot(dx, dy);
    }
  }
  if (retain_full_plane) {
    result.full_plane_log_likelihood = std::move(likelihood);
    result.likelihood_xy_mm = grid.xy_mm;
    result.likelihood_points = grid.points;
  }
  return result;
}

std::vector<double> sensitive_channel_xy(
    const Scene& scene, const std::vector<std::int32_t>& channel_ids) {
  std::map<std::int32_t, Vec2> positions;
  for (const auto& primitive : scene.mesh.analytic_primitives) {
    if (primitive.channel_id < 0) continue;
    const auto found = scene.surfaces.find(primitive.surface_id);
    if (found != scene.surfaces.end() &&
        found->second.kind == SurfaceKind::sensitive)
      positions[primitive.channel_id] = {
          primitive.center_mm.x, primitive.center_mm.y};
  }
  if (positions.size() < channel_ids.size()) {
    std::map<std::int32_t, std::pair<Vec2, std::uint64_t>> triangle_positions;
    for (std::size_t triangle = 0; triangle < scene.mesh.triangles.size();
         ++triangle) {
      const auto channel = scene.mesh.channel_id[triangle];
      if (channel < 0) continue;
      const auto& indices = scene.mesh.triangles[triangle];
      const auto& a = scene.mesh.vertices[indices[0]];
      const auto& b = scene.mesh.vertices[indices[1]];
      const auto& c = scene.mesh.vertices[indices[2]];
      auto& value = triangle_positions[channel];
      value.first.x += (a.x + b.x + c.x) / 3.0;
      value.first.y += (a.y + b.y + c.y) / 3.0;
      ++value.second;
    }
    for (const auto& [channel, value] : triangle_positions)
      if (value.second != 0 && positions.find(channel) == positions.end())
        positions[channel] = {
            value.first.x / value.second, value.first.y / value.second};
  }
  std::vector<double> result;
  result.reserve(channel_ids.size() * 2);
  for (const auto channel : channel_ids) {
    const auto found = positions.find(channel);
    if (found == positions.end())
      throw std::runtime_error("sensitive channel has no geometric center");
    result.push_back(found->second.x);
    result.push_back(found->second.y);
  }
  return result;
}

}  // namespace oos
