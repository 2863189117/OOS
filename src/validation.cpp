#include "oos/validation.hpp"

#include "oos/physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "oos/geometry.hpp"
#include "oos/analytic_geometry.hpp"
#include "oos/plugin.hpp"

namespace oos {
namespace {
Vec3 subtract(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}
double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
void error(ValidationReport& report, std::string code, std::string message) {
  report.issues.push_back(
      {ValidationIssue::Severity::error, std::move(code), std::move(message)});
}
using Edge = std::pair<std::uint32_t, std::uint32_t>;
Edge ordered_edge(std::uint32_t a, std::uint32_t b) {
  return std::minmax(a, b);
}

struct QuantizedVertex {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};
  bool operator==(const QuantizedVertex& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};
struct QuantizedHash {
  std::size_t operator()(const QuantizedVertex& value) const {
    std::size_t seed = std::hash<std::int64_t>{}(value.x);
    seed ^= std::hash<std::int64_t>{}(value.y) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<std::int64_t>{}(value.z) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};
}  // namespace

bool ValidationReport::ok() const {
  return std::none_of(issues.begin(), issues.end(), [](const auto& issue) {
    return issue.severity == ValidationIssue::Severity::error;
  });
}

void ValidationReport::throw_if_invalid() const {
  if (ok()) return;
  std::ostringstream message;
  message << "scene validation failed:";
  for (const auto& issue : issues) {
    if (issue.severity == ValidationIssue::Severity::error)
      message << "\n[" << issue.code << "] " << issue.message;
  }
  throw std::runtime_error(message.str());
}

ValidationReport SceneValidator::validate(const Scene& scene) {
  ValidationReport report;
  if (scene.schema_version != 1) error(report, "schema", "schema_version must be 1");
  if (!std::isfinite(scene.energy_eV) || scene.energy_eV <= 0.0)
    error(report, "energy", "energy_eV must be finite and positive");
  if (!std::isfinite(scene.numerics.geometry_tolerance_mm) ||
      scene.numerics.geometry_tolerance_mm <= 0.0)
    error(report, "geometry_tolerance",
          "geometry_tolerance_mm must be finite and positive");
  if (!std::isfinite(scene.numerics.ray_origin_offset_mm) ||
      scene.numerics.ray_origin_offset_mm < 0.0)
    error(report, "ray_origin_offset",
          "ray_origin_offset_mm must be finite and nonnegative");
  if (scene.numerics.lambertian_mu2_order == 0 ||
      scene.numerics.lambertian_phi_count == 0)
    error(report, "lambertian_quadrature",
          "Lambertian quadrature orders must be positive");
  if (!scene.media.count(scene.primary_domain))
    error(report, "primary_domain", "primary domain does not name a medium");
  if (scene.mesh.triangles.empty())
    error(report, "empty_geometry", "geometry contains no triangles");

  const std::size_t triangle_count = scene.mesh.triangles.size();
  const auto check_count = [&](std::size_t count, const char* name) {
    if (count != triangle_count)
      error(report, "shape", std::string(name) + " length must match triangles");
  };
  check_count(scene.mesh.surface_id.size(), "surface_id");
  if (!scene.mesh.surface_basis_id.empty())
    check_count(scene.mesh.surface_basis_id.size(), "surface_basis_id");
  check_count(scene.mesh.minus_domain_id.size(), "minus_domain_id");
  check_count(scene.mesh.plus_domain_id.size(), "plus_domain_id");
  check_count(scene.mesh.channel_id.size(), "channel_id");
  if (!scene.mesh.triangle_transport.empty())
    check_count(scene.mesh.triangle_transport.size(), "triangle_transport");
  if (!scene.mesh.triangle_source_quadrature.empty())
    check_count(scene.mesh.triangle_source_quadrature.size(),
                "triangle_source_quadrature");
  if (!report.ok()) return report;

  std::set<std::array<std::uint32_t, 3>> unique_triangles;
  std::map<std::pair<std::int32_t, Edge>, int> directed_domain_edges;
  std::vector<std::uint32_t> canonical_vertex(scene.mesh.vertices.size());
  std::unordered_map<QuantizedVertex, std::uint32_t, QuantizedHash>
      canonical_by_position;
  const double inverse_tolerance =
      1.0 / scene.numerics.geometry_tolerance_mm;
  for (std::uint32_t index = 0; index < scene.mesh.vertices.size(); ++index) {
    const auto& vertex = scene.mesh.vertices[index];
    const QuantizedVertex key{
        std::llround(vertex.x * inverse_tolerance),
        std::llround(vertex.y * inverse_tolerance),
        std::llround(vertex.z * inverse_tolerance)};
    const auto [found, inserted] =
        canonical_by_position.emplace(key, index);
    canonical_vertex[index] = found->second;
    (void)inserted;
  }
  for (std::size_t i = 0; i < triangle_count; ++i) {
    const auto triangle = scene.mesh.triangles[i];
    for (auto vertex : triangle) {
      if (vertex >= scene.mesh.vertices.size()) {
        error(report, "vertex_range", "triangle vertex lies outside vertices");
        break;
      }
    }
    if (!report.ok()) continue;
    auto identity = triangle;
    std::sort(identity.begin(), identity.end());
    if (!unique_triangles.insert(identity).second)
      error(report, "duplicate_triangle", "duplicate triangle found");
    const Vec3 a = scene.mesh.vertices[triangle[0]];
    const Vec3 b = scene.mesh.vertices[triangle[1]];
    const Vec3 c = scene.mesh.vertices[triangle[2]];
    if (0.5 * norm(cross(subtract(b, a), subtract(c, a))) <=
        scene.numerics.geometry_tolerance_mm *
            scene.numerics.geometry_tolerance_mm) {
      error(report, "degenerate_triangle", "triangle area is below tolerance");
    }
    const auto surface = scene.surfaces.find(scene.mesh.surface_id[i]);
    if (surface == scene.surfaces.end()) {
      error(report, "surface_reference", "triangle references undefined surface");
      continue;
    }
    const auto minus = scene.mesh.minus_domain_id[i];
    const auto plus = scene.mesh.plus_domain_id[i];
    if (minus == plus)
      error(report, "domain_adjacency", "triangle has the same domain on both sides");
    if (minus >= 0 && !scene.media.count(minus))
      error(report, "domain_reference", "minus domain is undefined");
    if (plus >= 0 && !scene.media.count(plus))
      error(report, "domain_reference", "plus domain is undefined");
    if (surface->second.kind == SurfaceKind::sensitive &&
        scene.mesh.channel_id[i] < 0)
      error(report, "sensitive_channel", "sensitive triangle lacks channel_id");
    if (surface->second.kind != SurfaceKind::sensitive &&
        surface->second.kind != SurfaceKind::custom_local &&
        scene.mesh.channel_id[i] >= 0)
      error(report, "unexpected_channel", "non-sensitive triangle has channel_id");
    const bool can_emit_lambertian =
        (surface->second.kind == SurfaceKind::lambertian &&
         surface->second.reflectivity > scene.numerics.energy_tolerance) ||
        surface->second.kind == SurfaceKind::custom_local ||
        (surface->second.kind == SurfaceKind::sensitive &&
         surface->second.remainder == RemainderAction::reflect_lambertian &&
         1.0 - surface->second.detection_probability >
             scene.numerics.energy_tolerance);
    if (!scene.mesh.surface_basis_id.empty() && can_emit_lambertian &&
        (minus == scene.primary_domain || plus == scene.primary_domain) &&
        scene.mesh.surface_basis_id[i] ==
            std::numeric_limits<std::uint32_t>::max())
      error(report, "surface_basis",
            "reflective triangle is absent from the active surface basis");

    for (int e = 0; e < 3; ++e) {
      const auto from = canonical_vertex[triangle[e]];
      const auto to = canonical_vertex[triangle[(e + 1) % 3]];
      const Edge edge = ordered_edge(from, to);
      if (minus >= 0)
        directed_domain_edges[{minus, edge}] += from < to ? 1 : -1;
      if (plus >= 0)
        directed_domain_edges[{plus, edge}] += from < to ? -1 : 1;
    }
  }
  for (const auto& [key, orientation] : directed_domain_edges) {
    if (orientation != 0) {
      error(report, "open_or_misoriented_domain",
            "domain boundary has an open or inconsistently oriented edge");
      break;
    }
  }

  for (const auto& [id, medium] : scene.media) {
    (void)id;
    if (!std::isfinite(medium.refractive_index) ||
        medium.refractive_index <= 0.0)
      error(report, "medium_index", "refractive index must be positive");
    if (!(medium.absorption_length_mm > 0.0))
      error(report, "medium_absorption", "absorption length must be positive");
  }
  for (const auto& [id, surface] : scene.surfaces) {
    (void)id;
    if (surface.reflectivity < 0.0 || surface.reflectivity > 1.0 ||
        surface.detection_probability < 0.0 ||
        surface.detection_probability > 1.0)
      error(report, "surface_probability", "surface probabilities must lie in [0,1]");
    if (!surface.behavior_declared)
      error(report, "surface_behavior",
            "surface must explicitly declare two_sided behavior and all "
            "model-specific probability fields");
    if (surface.kind == SurfaceKind::custom_local ||
        surface.kind == SurfaceKind::custom_nonlocal) {
      if (!surface.frame_declared) {
        error(report, "surface_frame",
              "custom surface requires a declared local_frame");
      } else {
        const auto& x = surface.frame_x;
        const auto& y = surface.frame_y;
        const auto& z = surface.frame_z;
        const auto& origin = surface.frame_origin_mm;
        const bool finite =
            std::isfinite(origin.x) && std::isfinite(origin.y) &&
            std::isfinite(origin.z) && std::isfinite(x.x) &&
            std::isfinite(x.y) && std::isfinite(x.z) &&
            std::isfinite(y.x) && std::isfinite(y.y) &&
            std::isfinite(y.z) && std::isfinite(z.x) &&
            std::isfinite(z.y) && std::isfinite(z.z);
        const double handedness = dot(cross(x, y), z);
        if (!finite || std::abs(norm(x) - 1.0) > 1.0e-12 ||
            std::abs(norm(y) - 1.0) > 1.0e-12 ||
            std::abs(norm(z) - 1.0) > 1.0e-12 ||
            std::abs(dot(x, y)) > 1.0e-12 ||
            std::abs(dot(x, z)) > 1.0e-12 ||
            std::abs(dot(y, z)) > 1.0e-12 ||
            std::abs(handedness - 1.0) > 1.0e-12)
          error(report, "surface_frame",
                "custom surface local_frame must be finite, orthonormal, "
                "and right-handed");
      }
      if (surface.plugin_path.empty()) {
        error(report, "plugin_path", "custom surface has no plugin path");
      } else {
        try {
          SurfacePlugin plugin(surface.plugin_path);
          plugin.validate(surface.plugin_config_json, scene.energy_eV);
          const bool expects_local =
              surface.kind == SurfaceKind::custom_local;
          if (expects_local !=
              (plugin.locality() == PluginLocality::local))
            error(report, "plugin_locality",
                  "scene surface locality disagrees with plugin ABI");
        } catch (const std::exception& exception) {
          error(report, "plugin_validation", exception.what());
        }
      }
      if (surface.kind == SurfaceKind::custom_nonlocal) {
        try {
          const auto config =
              nlohmann::json::parse(surface.plugin_config_json);
          const auto domain =
              config.at("nonlocal_domain_id").get<std::int32_t>();
          if (!scene.media.count(domain))
            error(report, "plugin_domain",
                  "nonlocal plugin domain is not a declared medium");
          for (std::size_t primitive = 0;
               primitive < scene.mesh.triangles.size(); ++primitive) {
            if (scene.mesh.surface_id[primitive] != surface.id) continue;
            const auto minus = scene.mesh.minus_domain_id[primitive];
            const auto plus = scene.mesh.plus_domain_id[primitive];
            if ((minus != domain && plus != domain) || minus < 0 || plus < 0) {
              error(report, "plugin_domain",
                    "nonlocal surface must separate its declared domain "
                    "from another medium");
              break;
            }
          }
        } catch (const std::exception&) {
          error(report, "plugin_domain",
                "nonlocal surface requires integer nonlocal_domain_id");
        }
      }
    }
  }
  for (const auto& primitive : scene.mesh.analytic_primitives) {
    try {
      validate_analytic_primitive(
          primitive,
          std::max(1.0e-12, scene.numerics.geometry_tolerance_mm));
    } catch (const std::exception& exception) {
      error(report, "analytic_geometry", exception.what());
      continue;
    }
    const auto surface = scene.surfaces.find(primitive.surface_id);
    if (surface == scene.surfaces.end()) {
      error(report, "surface_reference",
            "analytic primitive references undefined surface");
      continue;
    }
    if (primitive.minus_domain_id == primitive.plus_domain_id)
      error(report, "domain_adjacency",
            "analytic primitive has the same domain on both sides");
    if (primitive.minus_domain_id >= 0 &&
        !scene.media.count(primitive.minus_domain_id))
      error(report, "domain_reference",
            "analytic primitive minus domain is undefined");
    if (primitive.plus_domain_id >= 0 &&
        !scene.media.count(primitive.plus_domain_id))
      error(report, "domain_reference",
            "analytic primitive plus domain is undefined");
    if (surface->second.kind == SurfaceKind::sensitive &&
        primitive.channel_id < 0)
      error(report, "sensitive_channel",
            "sensitive analytic primitive lacks channel_id");
    if (surface->second.kind != SurfaceKind::sensitive &&
        surface->second.kind != SurfaceKind::custom_local &&
        primitive.channel_id >= 0)
      error(report, "unexpected_channel",
            "non-sensitive analytic primitive has channel_id");
  }
  for (const auto& element : scene.mesh.analytic_surface_elements) {
    if (element.primitive_index >= scene.mesh.analytic_primitives.size()) {
      error(report, "analytic_surface_basis",
            "analytic surface element references an unknown primitive");
      continue;
    }
    const auto& primitive =
        scene.mesh.analytic_primitives[element.primitive_index];
    if (!(element.area_mm2 > 0.0) ||
        !std::isfinite(element.area_mm2) ||
        !std::isfinite(element.center_mm.x) ||
        !std::isfinite(element.center_mm.y) ||
        !std::isfinite(element.center_mm.z) ||
        !std::isfinite(element.normal.x) ||
        !std::isfinite(element.normal.y) ||
        !std::isfinite(element.normal.z) ||
        std::abs(norm(element.normal) - 1.0) > 1.0e-10) {
      error(report, "analytic_surface_basis",
            "analytic surface element has invalid area, center, or normal");
      continue;
    }
    if (!(element.bounds[1] > element.bounds[0]) ||
        !(element.bounds[3] > element.bounds[2])) {
      error(report, "analytic_surface_basis",
            "analytic surface element coordinate bounds are empty");
    }
    const auto coordinate_matches =
        (element.coordinates == AnalyticSurfaceCoordinates::plane_uv &&
         (primitive.kind == GeometryPrimitiveKind::disk ||
          primitive.kind == GeometryPrimitiveKind::annulus ||
          primitive.kind == GeometryPrimitiveKind::perforated_disk)) ||
        (element.coordinates ==
             AnalyticSurfaceCoordinates::cylinder_phi_z &&
         primitive.kind == GeometryPrimitiveKind::finite_cylinder) ||
        (element.coordinates ==
             AnalyticSurfaceCoordinates::annulus_r2_phi &&
         (primitive.kind == GeometryPrimitiveKind::annulus ||
          primitive.kind == GeometryPrimitiveKind::disk)) ||
        (element.coordinates ==
             AnalyticSurfaceCoordinates::box_face_uv &&
         primitive.kind == GeometryPrimitiveKind::box);
    if (!coordinate_matches)
      error(report, "analytic_surface_basis",
            "analytic surface-element coordinates do not match primitive");
    const auto missing = std::numeric_limits<std::uint32_t>::max();
    const bool has_aperture_primitive =
        element.projected_aperture_primitive_index != missing;
    const bool has_aperture_hole =
        element.projected_aperture_hole_index != missing;
    if (has_aperture_primitive != has_aperture_hole) {
      error(report, "analytic_source_quadrature",
            "analytic surface element has an incomplete projected aperture");
    } else if (has_aperture_primitive) {
      if (element.projected_aperture_primitive_index >=
          scene.mesh.analytic_primitives.size()) {
        error(report, "analytic_source_quadrature",
              "projected aperture references an unknown primitive");
      } else {
        const auto& aperture = scene.mesh.analytic_primitives.at(
            element.projected_aperture_primitive_index);
        if (aperture.kind != GeometryPrimitiveKind::perforated_disk)
          error(report, "analytic_source_quadrature",
                "projected aperture must be a perforated disk");
        else if (element.projected_aperture_hole_index >=
                 aperture.holes.size())
          error(report, "analytic_source_quadrature",
                "projected aperture hole index is out of range");
      }
    }
  }
  if (!report.ok()) return report;

  try {
    Geometry geometry(scene);
    if (geometry.has_nonadjacent_self_intersection())
      error(report, "self_intersection",
            "non-adjacent triangles intersect each other");
    constexpr int ray_count = 128;
    constexpr double golden = 2.39996322972865332;
    for (int i = 0; i < ray_count; ++i) {
      const double z = 1.0 - 2.0 * (i + 0.5) / ray_count;
      const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
      const double phi = golden * i;
      const auto hit = geometry.intersect(
          {scene.primary_domain_seed_mm,
           {radius * std::cos(phi), radius * std::sin(phi), z}});
      if (!hit.valid) {
        error(report, "enclosure", "coverage ray escaped primary domain");
        break;
      }
      if (hit.minus_domain_id != scene.primary_domain &&
          hit.plus_domain_id != scene.primary_domain) {
        error(report, "enclosure",
              "coverage ray first hit is not adjacent to primary domain");
        break;
      }
    }
  } catch (const std::exception& exception) {
    error(report, "embree", exception.what());
  }
  return report;
}

}  // namespace oos
