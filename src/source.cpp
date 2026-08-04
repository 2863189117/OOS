#include "oos/source.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "oos/physics.hpp"

namespace oos {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

Vec3 vec3(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != 3)
    throw std::runtime_error("source vector must have three components");
  return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 subtract(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 scale(const Vec3& value, double factor) {
  return {factor * value.x, factor * value.y, factor * value.z};
}
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

std::pair<Vec3, Vec3> transverse_basis(const Vec3& axis) {
  const Vec3 normal = normalized(axis);
  const Vec3 trial =
      std::abs(normal.z) < 0.9 ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
  const Vec3 first = normalized(cross(trial, normal));
  return {first, cross(normal, first)};
}

double radical_inverse(std::uint64_t value, std::uint32_t base) {
  double result = 0.0;
  double scale_value = 1.0 / base;
  while (value) {
    result += (value % base) * scale_value;
    value /= base;
    scale_value /= base;
  }
  return result;
}

std::vector<std::pair<double, double>> gauss_legendre_unit_interval(
    std::uint32_t count) {
  if (count == 0)
    throw std::runtime_error("Gauss-Legendre source count is zero");
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
      if (count == 1u) value = root;
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

std::vector<std::pair<Vec3, double>> spatial_samples(const YAML::Node& node) {
  const std::string type = node["type"].as<std::string>();
  const std::uint32_t count = node["count"] ? node["count"].as<std::uint32_t>()
                                             : 1u;
  if (count == 0) throw std::runtime_error("source spatial count is zero");
  std::vector<std::pair<Vec3, double>> result;
  result.reserve(count);
  if (type == "point") {
    result.push_back({vec3(node["position_mm"]), 1.0});
  } else if (type == "line_segment") {
    const Vec3 start = vec3(node["start_mm"]);
    const Vec3 end = vec3(node["end_mm"]);
    const std::string quadrature =
        node["quadrature"] ? node["quadrature"].as<std::string>() : "midpoint";
    if (quadrature == "midpoint") {
      for (std::uint32_t i = 0; i < count; ++i) {
        const double u = (i + 0.5) / count;
        result.push_back({add(start, scale(subtract(end, start), u)),
                          1.0 / count});
      }
    } else if (quadrature == "gauss_legendre") {
      for (const auto& [u, weight] :
           gauss_legendre_unit_interval(count)) {
        result.push_back(
            {add(start, scale(subtract(end, start), u)), weight});
      }
    } else {
      throw std::runtime_error(
          "line_segment quadrature must be midpoint or gauss_legendre");
    }
  } else if (type == "disk" || type == "cylinder") {
    const Vec3 center = vec3(node["center_mm"]);
    const Vec3 axis = normalized(vec3(node["axis"]));
    const auto [u_axis, v_axis] = transverse_basis(axis);
    const double radius = node["radius_mm"].as<double>();
    const double length =
        type == "cylinder" ? node["length_mm"].as<double>() : 0.0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const double u = radical_inverse(i + 1, 2);
      const double phi = 2.0 * pi * radical_inverse(i + 1, 3);
      const double z =
          type == "cylinder"
              ? length * (radical_inverse(i + 1, 5) - 0.5)
              : 0.0;
      const Vec3 radial =
          add(scale(u_axis, radius * std::sqrt(u) * std::cos(phi)),
              scale(v_axis, radius * std::sqrt(u) * std::sin(phi)));
      result.push_back(
          {add(center, add(radial, scale(axis, z))), 1.0 / count});
    }
  } else if (type == "box") {
    const Vec3 minimum = vec3(node["min_mm"]);
    const Vec3 maximum = vec3(node["max_mm"]);
    for (std::uint32_t i = 0; i < count; ++i) {
      result.push_back(
          {{minimum.x + (maximum.x - minimum.x) *
                            radical_inverse(i + 1, 2),
            minimum.y + (maximum.y - minimum.y) *
                            radical_inverse(i + 1, 3),
            minimum.z + (maximum.z - minimum.z) *
                            radical_inverse(i + 1, 5)},
           1.0 / count});
    }
  } else if (type == "rectangular_line_neighborhood") {
    const auto points = uniform_rectangular_line_neighborhood_samples(
        vec3(node["line_center_mm"]),
        node["obstacle_half_width_mm"].as<double>(),
        node["obstacle_half_thickness_mm"].as<double>(),
        node["maximum_distance_mm"].as<double>(),
        node["medium_z_max_mm"].as<double>(), count);
    for (const auto& point : points)
      result.push_back({point, 1.0 / points.size()});
  } else if (type == "tetrahedron") {
    const auto vertices = node["vertices_mm"];
    if (!vertices.IsSequence() || vertices.size() != 4)
      throw std::runtime_error("tetrahedron needs four vertices");
    std::array<Vec3, 4> v{vec3(vertices[0]), vec3(vertices[1]),
                          vec3(vertices[2]), vec3(vertices[3])};
    for (std::uint32_t i = 0; i < count; ++i) {
      std::array<double, 4> exponential{};
      const std::array<std::uint32_t, 4> bases{2, 3, 5, 7};
      double total = 0.0;
      for (std::size_t j = 0; j < 4; ++j) {
        exponential[j] =
            -std::log(std::max(1e-15, radical_inverse(i + 1, bases[j])));
        total += exponential[j];
      }
      Vec3 point{};
      for (std::size_t j = 0; j < 4; ++j)
        point = add(point, scale(v[j], exponential[j] / total));
      result.push_back({point, 1.0 / count});
    }
  } else {
    throw std::runtime_error("unknown spatial source type " + type);
  }
  return result;
}

std::vector<std::pair<Vec3, double>> angular_samples(const YAML::Node& node) {
  const std::string type = node["type"].as<std::string>();
  if (type == "explicit") {
    const auto directions = node["directions"];
    std::vector<std::pair<Vec3, double>> result;
    double total = 0.0;
    for (const auto& entry : directions) {
      const double weight = entry["weight"].as<double>();
      if (!(weight >= 0.0)) throw std::runtime_error("negative direction weight");
      result.push_back({normalized(vec3(entry["direction"])), weight});
      total += weight;
    }
    if (!(total > 0.0)) throw std::runtime_error("direction weights sum to zero");
    for (auto& sample : result) sample.second /= total;
    return result;
  }
  if (type == "isotropic_product") {
    const std::uint32_t mu_order =
        node["mu_order"].as<std::uint32_t>();
    const std::uint32_t phi_count =
        node["phi_count"].as<std::uint32_t>();
    if (mu_order == 0 || phi_count == 0)
      throw std::runtime_error(
          "isotropic product quadrature orders must be positive");
    std::vector<std::pair<Vec3, double>> result;
    result.reserve(static_cast<std::size_t>(mu_order) * phi_count);
    for (const auto& [u, half_weight] :
         gauss_legendre_unit_interval(mu_order)) {
      const double mu = 2.0 * u - 1.0;
      const double transverse =
          std::sqrt(std::max(0.0, 1.0 - mu * mu));
      for (std::uint32_t phi_index = 0; phi_index < phi_count;
           ++phi_index) {
        const double phi =
            2.0 * pi * (phi_index + 0.5) / phi_count;
        result.push_back(
            {{transverse * std::cos(phi), transverse * std::sin(phi), mu},
             half_weight / phi_count});
      }
    }
    return result;
  }
  const std::uint32_t count = node["count"].as<std::uint32_t>();
  if (count == 0) throw std::runtime_error("source angular count is zero");
  const Vec3 axis =
      node["axis"] ? normalized(vec3(node["axis"])) : Vec3{0, 0, 1};
  const auto [u_axis, v_axis] = transverse_basis(axis);
  const double cone_cosine =
      type == "cone"
          ? std::cos(node["half_angle_deg"].as<double>() * pi / 180.0)
          : -1.0;
  std::vector<std::pair<Vec3, double>> result;
  result.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    const double u = (i + 0.5) / count;
    const double phi = 2.0 * pi * radical_inverse(i + 1, 2);
    double mu{};
    if (type == "isotropic")
      mu = 1.0 - 2.0 * u;
    else if (type == "cosine")
      mu = std::sqrt(u);
    else if (type == "cone")
      mu = cone_cosine + (1.0 - cone_cosine) * u;
    else
      throw std::runtime_error("unknown angular source type " + type);
    const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    result.push_back(
        {normalized(add(scale(axis, mu),
                        add(scale(u_axis, transverse * std::cos(phi)),
                            scale(v_axis, transverse * std::sin(phi))))),
         1.0 / count});
  }
  return result;
}

}  // namespace

std::vector<Vec3> uniform_rectangular_line_neighborhood_samples(
    const Vec3& line_center_mm, double obstacle_half_width_mm,
    double obstacle_half_thickness_mm, double maximum_distance_mm,
    double medium_z_max_mm, std::uint32_t count) {
  if (!std::isfinite(line_center_mm.x) ||
      !std::isfinite(line_center_mm.y) ||
      !std::isfinite(line_center_mm.z) ||
      !(obstacle_half_width_mm > 0.0) ||
      !(obstacle_half_thickness_mm > 0.0) ||
      !(maximum_distance_mm > 0.0) ||
      !std::isfinite(medium_z_max_mm) || count == 0)
    throw std::runtime_error("line-neighborhood parameters are invalid");
  const double y_min =
      line_center_mm.y - obstacle_half_width_mm - maximum_distance_mm;
  const double y_max =
      line_center_mm.y + obstacle_half_width_mm + maximum_distance_mm;
  const double z_min =
      line_center_mm.z - obstacle_half_thickness_mm - maximum_distance_mm;
  const double z_max = std::min(
      medium_z_max_mm,
      line_center_mm.z + obstacle_half_thickness_mm + maximum_distance_mm);
  if (!(z_max > z_min))
    throw std::runtime_error("line neighborhood has no medium extent");
  std::vector<Vec3> result;
  result.reserve(count);
  const std::uint64_t maximum_trials =
      std::max<std::uint64_t>(10000, 1000 * count);
  for (std::uint64_t trial = 1;
       trial <= maximum_trials && result.size() < count; ++trial) {
    const double y =
        y_min + (y_max - y_min) * radical_inverse(trial, 2);
    const double z =
        z_min + (z_max - z_min) * radical_inverse(trial, 3);
    const double dy =
        std::max(std::abs(y - line_center_mm.y) - obstacle_half_width_mm,
                 0.0);
    const double dz =
        std::max(std::abs(z - line_center_mm.z) -
                     obstacle_half_thickness_mm,
                 0.0);
    const bool inside_metal =
        std::abs(y - line_center_mm.y) < obstacle_half_width_mm &&
        std::abs(z - line_center_mm.z) < obstacle_half_thickness_mm;
    if (!inside_metal &&
        std::hypot(dy, dz) <= maximum_distance_mm)
      result.push_back({line_center_mm.x, y, z});
  }
  if (result.size() != count)
    throw std::runtime_error(
        "line-neighborhood rejection sampler did not converge");
  return result;
}

std::vector<SourceQuadrature> load_sources_yaml(
    const std::filesystem::path& path, const Scene& scene) {
  const YAML::Node root = YAML::LoadFile(path.string());
  if (root["schema_version"].as<std::uint32_t>() != 1)
    throw std::runtime_error("sources schema_version must be 1");
  std::vector<SourceQuadrature> result;
  for (const auto& source : root["sources"]) {
    SourceQuadrature quadrature;
    quadrature.id = source["id"].as<std::string>();
    const auto positions = spatial_samples(source["spatial"]);
    const std::int32_t domain =
        source["domain"] ? source["domain"].as<std::int32_t>()
                         : scene.primary_domain;
    const Stokes polarization =
        source["stokes"]
            ? Stokes{source["stokes"][0].as<double>(),
                     source["stokes"][1].as<double>(),
                     source["stokes"][2].as<double>(),
                     source["stokes"][3].as<double>()}
            : Stokes{1, 0, 0, 0};
    const auto angular = source["angular"];
    const std::string angular_type =
        angular["type"].as<std::string>();
    if (angular_type == "isotropic_surface_shape_factor") {
      quadrature.integration =
          SourceIntegration::isotropic_surface_shape_factor;
      if (angular["relative_tolerance"])
        quadrature.shape_factor.relative_tolerance =
            angular["relative_tolerance"].as<double>();
      if (angular["maximum_subdivision_depth"])
        quadrature.shape_factor.maximum_subdivision_depth =
            angular["maximum_subdivision_depth"].as<std::uint32_t>();
      if (angular["minimum_refinement_solid_angle_fraction"])
        quadrature.shape_factor.minimum_refinement_solid_angle_fraction =
            angular["minimum_refinement_solid_angle_fraction"].as<double>();
      if (angular["minimum_feature_solid_angle_fraction"])
        quadrature.shape_factor.minimum_feature_solid_angle_fraction =
            angular["minimum_feature_solid_angle_fraction"].as<double>();
      if (angular["feature_dihedral_degrees"])
        quadrature.shape_factor.feature_dihedral_degrees =
            angular["feature_dihedral_degrees"].as<double>();
      if (angular["maximum_approximate_solid_angle_fraction"])
        quadrature.shape_factor.maximum_approximate_solid_angle_fraction =
            angular["maximum_approximate_solid_angle_fraction"].as<double>();
      if (angular["aperture_edge_phi_order"])
        quadrature.shape_factor.aperture_edge_phi_order =
            angular["aperture_edge_phi_order"].as<std::uint32_t>();
      if (angular["aperture_edge_weight_threshold"])
        quadrature.shape_factor.aperture_edge_weight_threshold =
            angular["aperture_edge_weight_threshold"].as<double>();
      if (!(quadrature.shape_factor.relative_tolerance > 0.0) ||
          !std::isfinite(quadrature.shape_factor.relative_tolerance))
        throw std::runtime_error(
            "shape-factor relative_tolerance must be finite and positive");
      if (!(quadrature.shape_factor.minimum_refinement_solid_angle_fraction >
            0.0) ||
          !std::isfinite(
              quadrature.shape_factor
                  .minimum_refinement_solid_angle_fraction) ||
          !(quadrature.shape_factor.minimum_feature_solid_angle_fraction >
            0.0) ||
          !std::isfinite(
              quadrature.shape_factor.minimum_feature_solid_angle_fraction))
        throw std::runtime_error(
            "shape-factor refinement fractions must be finite and positive");
      if (!(quadrature.shape_factor.feature_dihedral_degrees >= 0.0) ||
          !(quadrature.shape_factor.feature_dihedral_degrees < 180.0) ||
          !std::isfinite(
              quadrature.shape_factor.feature_dihedral_degrees))
        throw std::runtime_error(
            "shape-factor feature_dihedral_degrees must be in [0,180)");
      if (!(quadrature.shape_factor
                .maximum_approximate_solid_angle_fraction >= 0.0) ||
          !std::isfinite(
              quadrature.shape_factor
                  .maximum_approximate_solid_angle_fraction))
        throw std::runtime_error(
            "shape-factor maximum_approximate_solid_angle_fraction must "
            "be finite and nonnegative");
      if (quadrature.shape_factor.aperture_edge_phi_order == 0)
        throw std::runtime_error(
            "shape-factor aperture_edge_phi_order must be positive");
      if (!(quadrature.shape_factor.aperture_edge_weight_threshold >= 0.0) ||
          !std::isfinite(
              quadrature.shape_factor.aperture_edge_weight_threshold))
        throw std::runtime_error(
            "shape-factor aperture edge threshold must be finite and "
            "non-negative");
      quadrature.points.reserve(positions.size());
      for (const auto& [position, position_weight] : positions) {
        Stokes weighted = polarization;
        weighted.i *= position_weight;
        weighted.q *= position_weight;
        weighted.u *= position_weight;
        weighted.v *= position_weight;
        quadrature.points.push_back({position, weighted, domain});
      }
      result.push_back(std::move(quadrature));
      continue;
    }
    const auto directions = angular_samples(angular);
    for (const auto& [position, position_weight] : positions) {
      for (const auto& [direction, direction_weight] : directions) {
        const auto [reference, ignored] = transverse_basis(direction);
        (void)ignored;
        Stokes weighted = polarization;
        const double weight = position_weight * direction_weight;
        weighted.i *= weight;
        weighted.q *= weight;
        weighted.u *= weight;
        weighted.v *= weight;
        quadrature.rays.push_back(
            {{position, direction}, weighted, reference, domain});
      }
    }
    result.push_back(std::move(quadrature));
  }
  if (result.empty()) throw std::runtime_error("sources.yaml has no sources");
  return result;
}

}  // namespace oos
