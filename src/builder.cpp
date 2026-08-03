#include "oos/builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <limits>

#include <nlohmann/json.hpp>

#include "oos/analytic_geometry.hpp"
#include "oos/geometry.hpp"
#include "oos/function_operator.hpp"
#include "oos/hash.hpp"
#include "oos/physics.hpp"
#include "oos/plugin.hpp"
#include "oos/surface_behavior.hpp"
#include "oos/validation.hpp"

namespace oos {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::vector<std::pair<double, double>> gauss_legendre_unit_interval(
    std::uint32_t count) {
  if (count == 0)
    throw std::runtime_error("Gauss-Legendre quadrature order is zero");
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

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 subtract(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 scale(const Vec3& value, double factor) {
  return {factor * value.x, factor * value.y, factor * value.z};
}
double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double triangle_area(const Scene& scene, std::uint32_t primitive) {
  const auto triangle = scene.mesh.triangles.at(primitive);
  return 0.5 * norm(cross(
                   subtract(scene.mesh.vertices[triangle[1]],
                            scene.mesh.vertices[triangle[0]]),
                   subtract(scene.mesh.vertices[triangle[2]],
                            scene.mesh.vertices[triangle[0]])));
}

Vec3 triangle_normal(const Scene& scene, std::uint32_t primitive) {
  const auto triangle = scene.mesh.triangles.at(primitive);
  return normalized(cross(subtract(scene.mesh.vertices[triangle[1]],
                                   scene.mesh.vertices[triangle[0]]),
                          subtract(scene.mesh.vertices[triangle[2]],
                                   scene.mesh.vertices[triangle[0]])));
}

Vec3 triangle_center(const Scene& scene, std::uint32_t primitive) {
  const auto triangle = scene.mesh.triangles.at(primitive);
  return scale(add(add(scene.mesh.vertices[triangle[0]],
                       scene.mesh.vertices[triangle[1]]),
                   scene.mesh.vertices[triangle[2]]),
               1.0 / 3.0);
}

struct WeightedRay {
  Ray ray;
  Stokes stokes;
  Vec3 reference_axis;
  std::int32_t domain{};
  std::uint32_t depth{};
  std::optional<Hit> known_first_hit;
};

struct RowAccumulation {
  std::map<std::uint32_t, double> transition;
  std::map<std::uint32_t, double> detection;
  std::map<std::uint32_t, double> losses;
};

struct SurfaceEmitter {
  std::uint64_t geometry_key{};
  Vec3 center;
  Vec3 normal;
  Vec3 tangent;
  double area_mm2{};
  std::uint32_t surface_id{};
  std::uint32_t surface_basis_id{};
  std::int32_t minus_domain_id{-1};
  std::int32_t plus_domain_id{-1};
  std::string label;
};

struct LocalSurfaceBasis {
  std::vector<std::vector<SurfaceEmitter>> state_emitters;
  std::unordered_map<std::uint64_t, std::uint32_t> primitive_to_state;
  std::vector<std::string> state_labels;
};

bool owns_lambertian_state(const SurfaceModel& surface, double tolerance) {
  if (surface.kind == SurfaceKind::lambertian)
    return surface.reflectivity > tolerance;
  if (surface.kind == SurfaceKind::custom_local) return true;
  return surface.kind == SurfaceKind::sensitive &&
         surface.remainder == RemainderAction::reflect_lambertian &&
         (1.0 - surface.detection_probability) > tolerance;
}

LocalSurfaceBasis make_local_surface_basis(const Scene& scene) {
  using Key = std::tuple<std::uint32_t, std::uint32_t, bool>;
  std::map<Key, std::uint32_t> key_to_state;
  LocalSurfaceBasis result;
  const auto append =
      [&](SurfaceEmitter emitter, bool primary_is_minus) {
        const Key key{emitter.surface_id, emitter.surface_basis_id,
                      primary_is_minus};
        auto [entry, inserted] =
            key_to_state.emplace(key, result.state_emitters.size());
        if (inserted) result.state_emitters.emplace_back();
        const auto state = entry->second;
        result.primitive_to_state.emplace(emitter.geometry_key, state);
        result.state_emitters.at(state).push_back(std::move(emitter));
      };
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    if (!scene.mesh.triangle_transport.empty() &&
        !scene.mesh.triangle_transport.at(primitive))
      continue;
    const auto& surface = scene.surfaces.at(scene.mesh.surface_id[primitive]);
    const bool primary_is_minus =
        scene.mesh.minus_domain_id[primitive] == scene.primary_domain;
    const bool primary_is_plus =
        scene.mesh.plus_domain_id[primitive] == scene.primary_domain;
    if (!owns_lambertian_state(surface, scene.numerics.energy_tolerance) ||
        (!primary_is_minus && !primary_is_plus))
      continue;
    const std::uint32_t basis_id =
        scene.mesh.surface_basis_id.empty()
            ? primitive
            : scene.mesh.surface_basis_id.at(primitive);
    const auto triangle = scene.mesh.triangles.at(primitive);
    append(
        {static_cast<std::uint64_t>(primitive),
         triangle_center(scene, primitive),
         triangle_normal(scene, primitive),
         normalized(subtract(scene.mesh.vertices.at(triangle[1]),
                             scene.mesh.vertices.at(triangle[0]))),
         triangle_area(scene, primitive),
         surface.id,
         basis_id,
         scene.mesh.minus_domain_id.at(primitive),
         scene.mesh.plus_domain_id.at(primitive),
         "primitive:" + std::to_string(primitive)},
        primary_is_minus);
  }
  for (std::uint32_t element_index = 0;
       element_index < scene.mesh.analytic_surface_elements.size();
       ++element_index) {
    const auto& element =
        scene.mesh.analytic_surface_elements.at(element_index);
    const auto& primitive =
        scene.mesh.analytic_primitives.at(element.primitive_index);
    const auto& surface = scene.surfaces.at(primitive.surface_id);
    const bool primary_is_minus =
        primitive.minus_domain_id == scene.primary_domain;
    const bool primary_is_plus =
        primitive.plus_domain_id == scene.primary_domain;
    if (!owns_lambertian_state(surface, scene.numerics.energy_tolerance) ||
        (!primary_is_minus && !primary_is_plus))
      continue;
    Vec3 tangent = primitive.axis_x;
    if (element.coordinates ==
        AnalyticSurfaceCoordinates::cylinder_phi_z) {
      const double phi =
          0.5 * (element.bounds[0] + element.bounds[1]);
      tangent = normalized(
          add(scale(primitive.axis_x, -std::sin(phi)),
              scale(primitive.axis_y, std::cos(phi))));
    }
    append(
        {analytic_surface_element_geometry_key(element_index),
         element.center_mm,
         element.normal,
         tangent,
         element.area_mm2,
         primitive.surface_id,
         element.surface_basis_id,
         primitive.minus_domain_id,
         primitive.plus_domain_id,
         "element:" + std::to_string(element.surface_element)},
        primary_is_minus);
  }
  result.state_labels.reserve(result.state_emitters.size());
  for (const auto& emitters : result.state_emitters) {
    const auto& first = emitters.front();
    const auto& surface = scene.surfaces.at(first.surface_id);
    if (emitters.size() == 1)
      result.state_labels.push_back(surface.name + "/" + first.label);
    else
      result.state_labels.push_back(
          surface.name + "/basis:" +
          std::to_string(first.surface_basis_id));
  }
  return result;
}

struct CustomSurfaceRuntime {
  std::shared_ptr<SurfacePlugin> plugin;
  std::string config_json;
  PluginBuildResult payload;
  std::uint64_t state_offset{};
  std::uint64_t state_count{};
  std::int32_t nonlocal_domain_id{-1};
  bool functional{};
  oos_function_operator_descriptor_v1 function_descriptor{};
};

using CustomRuntimeMap =
    std::unordered_map<std::uint32_t, CustomSurfaceRuntime>;

oos_surface_hit_v3 plugin_hit(const Scene& scene, const SurfaceModel& surface,
                              const WeightedRay& ray, const Hit& hit,
                              const Vec3& point,
                              std::int32_t transmitted_domain) {
  oos_surface_hit_v3 result{};
  result.surface_element = hit.surface_element;
  std::copy(hit.barycentric.begin(), hit.barycentric.end(),
            result.barycentric);
  result.incident_side =
      ray.domain == hit.minus_domain_id ? 0u : 1u;
  const Vec3 local_point = subtract(point, surface.frame_origin_mm);
  result.point_local_mm[0] = dot(local_point, surface.frame_x);
  result.point_local_mm[1] = dot(local_point, surface.frame_y);
  result.point_local_mm[2] = dot(local_point, surface.frame_z);
  const auto direction = normalized(ray.ray.direction);
  result.direction_local[0] = dot(direction, surface.frame_x);
  result.direction_local[1] = dot(direction, surface.frame_y);
  result.direction_local[2] = dot(direction, surface.frame_z);
  result.reference_axis_local[0] =
      dot(ray.reference_axis, surface.frame_x);
  result.reference_axis_local[1] =
      dot(ray.reference_axis, surface.frame_y);
  result.reference_axis_local[2] =
      dot(ray.reference_axis, surface.frame_z);
  result.stokes[0] = ray.stokes.i;
  result.stokes[1] = ray.stokes.q;
  result.stokes[2] = ray.stokes.u;
  result.stokes[3] = ray.stokes.v;
  result.incident_refractive_index =
      scene.media.at(ray.domain).refractive_index;
  result.transmitted_refractive_index =
      transmitted_domain >= 0
          ? scene.media.at(transmitted_domain).refractive_index
          : result.incident_refractive_index;
  return result;
}

double signed_basis_angle(const Vec3& from, const Vec3& to,
                          const Vec3& direction) {
  return std::atan2(dot(cross(from, to), direction), dot(from, to));
}

void apply_local_branches(
    const Scene& scene, std::uint64_t geometry_key, std::int32_t channel,
    const Vec3& point,
    const Vec3& direction, const Vec3& toward_incident, const Vec3& s_axis,
    std::int32_t incident_domain, std::int32_t transmitted_domain,
    std::uint32_t depth, double nudge,
    const std::unordered_map<std::uint64_t, std::uint32_t>&
        primitive_to_state,
    const std::unordered_map<std::int32_t, std::uint32_t>& channel_to_column,
    const std::vector<LocalSurfaceBranch>& branches,
    std::vector<WeightedRay>& pending, RowAccumulation& output) {
  for (const auto& branch : branches) {
    switch (branch.kind) {
      case LocalBranchKind::specular_reflection: {
        const Vec3 reflected = reflect(direction, toward_incident);
        pending.push_back(
            {{add(point, scale(toward_incident, nudge)), reflected},
             branch.stokes, s_axis, incident_domain, depth + 1,
             std::nullopt});
        break;
      }
      case LocalBranchKind::specular_transmission: {
        if (transmitted_domain < 0)
          throw std::runtime_error("surface transmits outside the scene");
        const auto transmitted = refract(
            direction, toward_incident,
            scene.media.at(incident_domain).refractive_index,
            scene.media.at(transmitted_domain).refractive_index);
        if (!transmitted)
          throw std::runtime_error(
              "surface requested transmission during total internal "
              "reflection");
        pending.push_back(
            {{add(point, scale(toward_incident, -nudge)), *transmitted},
             branch.stokes, s_axis, transmitted_domain, depth + 1,
             std::nullopt});
        break;
      }
      case LocalBranchKind::straight_transmission:
        if (transmitted_domain < 0) {
          output.losses[2] += branch.stokes.i;
        } else {
          pending.push_back(
              {{add(point, scale(toward_incident, -nudge)), direction},
               branch.stokes, s_axis, transmitted_domain, depth + 1,
               std::nullopt});
        }
        break;
      case LocalBranchKind::lambertian_reflection: {
        const auto state = primitive_to_state.find(geometry_key);
        if (state == primitive_to_state.end())
          throw std::runtime_error(
              "Lambertian branch has no geometry-owned state: geometry_key=" +
              std::to_string(geometry_key));
        output.transition[state->second] += branch.stokes.i;
        break;
      }
      case LocalBranchKind::detection: {
        if (channel < 0)
          throw std::runtime_error("detecting surface has no channel");
        output.detection[channel_to_column.at(channel)] += branch.stokes.i;
        break;
      }
      case LocalBranchKind::absorption:
        output.losses[1] += branch.stokes.i;
        break;
    }
  }
}

void trace_branch(const Scene& scene, const Geometry& geometry,
                  const std::unordered_map<std::uint64_t, std::uint32_t>&
                      primitive_to_state,
                  const std::unordered_map<std::int32_t, std::uint32_t>&
                      channel_to_column,
                  const CustomRuntimeMap& custom_runtimes,
                  const WeightedRay& input, RowAccumulation& output) {
  std::vector<WeightedRay> pending{input};
  while (!pending.empty()) {
    WeightedRay ray = pending.back();
    pending.pop_back();
    if (ray.stokes.i <= scene.numerics.energy_tolerance) {
      output.losses[3] += ray.stokes.i;
      continue;
    }
    if (ray.depth >= scene.numerics.maximum_specular_hits) {
      output.losses[3] += ray.stokes.i;
      continue;
    }
    const Hit hit = ray.known_first_hit
                        ? *ray.known_first_hit
                        : geometry.intersect(ray.ray, ray.domain);
    if (!hit.valid) {
      // Scene validation has already established that the traced domain is
      // enclosed.  A no-hit result is therefore kept separate from physical
      // transmission to the exterior: Embree traces float32 rays, so grazing
      // seams can lose a small amount of weight even for a closed float64
      // input mesh.
      output.losses[4] += ray.stokes.i;
      continue;
    }
    const auto& surface = scene.surfaces.at(hit.surface_id);
    const auto primitive = hit.primitive_id;
    const Vec3 point =
        add(ray.ray.origin, scale(normalized(ray.ray.direction), hit.distance));
    const auto& propagation_medium = scene.media.at(ray.domain);
    const double survival =
        std::isfinite(propagation_medium.absorption_length_mm)
            ? std::exp(-hit.distance /
                       propagation_medium.absorption_length_mm)
            : 1.0;
    output.losses[0] += ray.stokes.i * (1.0 - survival);
    ray.stokes = {ray.stokes.i * survival, ray.stokes.q * survival,
                  ray.stokes.u * survival, ray.stokes.v * survival};
    if (surface.kind == SurfaceKind::specular_reflector ||
        surface.kind == SurfaceKind::lambertian ||
        surface.kind == SurfaceKind::sensitive) {
      const auto minus = hit.minus_domain_id;
      const auto plus = hit.plus_domain_id;
      const bool incident_from_minus = ray.domain == minus;
      if (!incident_from_minus && ray.domain != plus)
        throw std::runtime_error(
            "ray domain is not adjacent to surface: domain=" +
            std::to_string(ray.domain) +
            " surface=" + std::to_string(surface.id) +
            " primitive=" + std::to_string(primitive) +
            " minus=" + std::to_string(minus) +
            " plus=" + std::to_string(plus) +
            " origin=(" + std::to_string(ray.ray.origin.x) + "," +
            std::to_string(ray.ray.origin.y) + "," +
            std::to_string(ray.ray.origin.z) + ")" +
            " direction=(" + std::to_string(ray.ray.direction.x) + "," +
            std::to_string(ray.ray.direction.y) + "," +
            std::to_string(ray.ray.direction.z) + ")");
      const auto transmitted_domain = incident_from_minus ? plus : minus;
      const auto normal = hit.normal;
      const auto toward_incident =
          incident_from_minus ? scale(normal, -1.0) : normal;
      const auto direction = normalized(ray.ray.direction);
      const auto branches = evaluate_builtin_surface(
          surface, {ray.stokes,
                    scene.media.at(ray.domain).refractive_index,
                    transmitted_domain >= 0
                        ? scene.media.at(transmitted_domain).refractive_index
                        : scene.media.at(ray.domain).refractive_index,
                    std::max(0.0, -dot(direction, toward_incident))});
      apply_local_branches(
          scene, hit.geometry_key, hit.channel_id, point, direction,
          toward_incident,
          ray.reference_axis, ray.domain, transmitted_domain, ray.depth,
          geometry.ray_origin_offset_mm(), primitive_to_state,
          channel_to_column, branches, pending, output);
      continue;
    }
    if (surface.kind == SurfaceKind::custom_local) {
      const auto runtime = custom_runtimes.find(surface.id);
      if (runtime == custom_runtimes.end())
        throw std::runtime_error("local custom surface has no runtime");
      const auto minus = hit.minus_domain_id;
      const auto plus = hit.plus_domain_id;
      const std::int32_t transmitted_domain =
          ray.domain == minus ? plus : minus;
      const Vec3 normal = hit.normal;
      const auto interaction = runtime->second.plugin->interact_local(
          runtime->second.config_json, scene.energy_eV,
          plugin_hit(scene, surface, ray, hit, point, transmitted_domain));
      const Vec3 direction = normalized(ray.ray.direction);
      const Vec3 toward_incident =
          ray.domain == minus ? scale(normal, -1.0) : normal;
      const auto branches = evaluate_custom_local_surface(
          interaction,
          {ray.stokes, scene.media.at(ray.domain).refractive_index,
           transmitted_domain >= 0
               ? scene.media.at(transmitted_domain).refractive_index
               : scene.media.at(ray.domain).refractive_index,
           std::max(0.0, -dot(direction, toward_incident))},
          scene.numerics.energy_tolerance);
      apply_local_branches(
          scene, hit.geometry_key, hit.channel_id, point, direction,
          toward_incident,
          ray.reference_axis, ray.domain, transmitted_domain, ray.depth,
          geometry.ray_origin_offset_mm(), primitive_to_state,
          channel_to_column, branches, pending, output);
      continue;
    }
    if (surface.kind == SurfaceKind::custom_nonlocal) {
      const auto runtime = custom_runtimes.find(surface.id);
      if (runtime == custom_runtimes.end())
        throw std::runtime_error("nonlocal custom surface has no runtime");
      const auto minus = hit.minus_domain_id;
      const auto plus = hit.plus_domain_id;
      const bool incident_from_minus = ray.domain == minus;
      if (!incident_from_minus && ray.domain != plus)
        throw std::runtime_error("ray domain is not adjacent to custom interface");
      const std::int32_t transmitted_domain =
          incident_from_minus ? plus : minus;
      if (transmitted_domain < 0)
        throw std::runtime_error(
            "nonlocal custom surface must separate two media");
      if (transmitted_domain != runtime->second.nonlocal_domain_id)
        throw std::runtime_error(
            "generic rays may enter, but may not originate inside, a "
            "nonlocal custom domain");
      const Vec3 normal = hit.normal;
      const Vec3 toward_incident =
          incident_from_minus ? scale(normal, -1.0) : normal;
      const Vec3 direction = normalized(ray.ray.direction);
      Vec3 s_axis = cross(direction, toward_incident);
      if (norm(s_axis) < 1e-14) s_axis = ray.reference_axis;
      s_axis = normalized(s_axis);
      const Stokes interface_basis =
          rotate_stokes(ray.stokes,
                        signed_basis_angle(ray.reference_axis, s_axis, direction));
      SurfaceModel dielectric;
      dielectric.kind = SurfaceKind::dielectric_fresnel;
      const auto interface_branches = evaluate_builtin_surface(
          dielectric,
          {interface_basis,
           scene.media.at(ray.domain).refractive_index,
           scene.media.at(transmitted_domain).refractive_index,
           std::max(0.0, -dot(direction, toward_incident))});
      for (const auto& branch : interface_branches) {
        if (branch.kind == LocalBranchKind::specular_reflection) {
          const Vec3 reflected_direction = reflect(direction, toward_incident);
          pending.push_back(
              {{add(point, scale(toward_incident,
                                 geometry.ray_origin_offset_mm())),
                reflected_direction},
               branch.stokes, s_axis, ray.domain, ray.depth + 1,
               std::nullopt});
          continue;
        }
        if (branch.kind != LocalBranchKind::specular_transmission)
          throw std::runtime_error(
              "dielectric entry behavior returned an invalid branch");
        const auto transmitted_direction =
            refract(direction, toward_incident,
                    scene.media.at(ray.domain).refractive_index,
                    scene.media.at(transmitted_domain).refractive_index);
        if (transmitted_direction &&
            branch.stokes.i > scene.numerics.energy_tolerance) {
          WeightedRay transmitted_ray{
              {add(point, scale(toward_incident,
                                -geometry.ray_origin_offset_mm())),
              *transmitted_direction},
              branch.stokes, s_axis, transmitted_domain, ray.depth + 1,
              std::nullopt};
          const auto deposition = runtime->second.plugin->deposit_nonlocal(
              runtime->second.config_json, scene.energy_eV,
              plugin_hit(scene, surface, transmitted_ray, hit, point,
                         transmitted_domain));
          if (deposition.weights.empty())
            throw std::runtime_error(
                "nonlocal custom surface returned an empty deposition");
          double deposited = 0.0;
          for (std::size_t i = 0; i < deposition.weights.size(); ++i) {
            const auto weight = deposition.weights[i];
            const auto state = deposition.state_indices[i];
            if (!std::isfinite(weight) || weight < 0.0 ||
                state >= runtime->second.state_count)
              throw std::runtime_error(
                  "nonlocal custom surface returned an invalid deposition");
            deposited += weight;
            output.transition[static_cast<std::uint32_t>(
                runtime->second.state_offset + state)] +=
                branch.stokes.i * weight;
          }
          if (std::abs(deposited - 1.0) >
              scene.numerics.energy_tolerance)
            throw std::runtime_error(
                "nonlocal custom surface deposition is not conservative");
        }
      }
      continue;
    }

    const auto minus = hit.minus_domain_id;
    const auto plus = hit.plus_domain_id;
    const bool incident_from_minus = ray.domain == minus;
    if (!incident_from_minus && ray.domain != plus)
      throw std::runtime_error("ray domain is not adjacent to interface");
    const std::int32_t transmitted_domain =
        incident_from_minus ? plus : minus;
    if (transmitted_domain < 0) {
      output.losses[2] += ray.stokes.i;
      continue;
    }
    const Vec3 geometric_normal = hit.normal;
    const Vec3 toward_incident =
        incident_from_minus ? scale(geometric_normal, -1.0) : geometric_normal;
    const Vec3 direction = normalized(ray.ray.direction);
    Vec3 s_axis = cross(direction, toward_incident);
    if (norm(s_axis) < 1e-14) s_axis = ray.reference_axis;
    s_axis = normalized(s_axis);
    const Stokes interface_basis =
        rotate_stokes(ray.stokes,
                      signed_basis_angle(ray.reference_axis, s_axis, direction));
    const auto& incident_medium = scene.media.at(ray.domain);
    const auto& transmitted_medium = scene.media.at(transmitted_domain);
    const double cos_i = std::max(0.0, -dot(direction, toward_incident));
    const auto branches = evaluate_builtin_surface(
        surface, {interface_basis, incident_medium.refractive_index,
                  transmitted_medium.refractive_index, cos_i});
    apply_local_branches(
        scene, hit.geometry_key, hit.channel_id, point, direction,
        toward_incident, s_axis,
        ray.domain, transmitted_domain, ray.depth,
        geometry.ray_origin_offset_mm(), primitive_to_state, channel_to_column,
        branches, pending, output);
  }
}

double row_total(const RowAccumulation& row) {
  double result = 0.0;
  for (const auto& [column, value] : row.transition) {
    (void)column;
    result += value;
  }
  for (const auto& [column, value] : row.detection) {
    (void)column;
    result += value;
  }
  for (const auto& [column, value] : row.losses) {
    (void)column;
    result += value;
  }
  return result;
}

void add_row(RowAccumulation& target, const RowAccumulation& source,
             double factor = 1.0) {
  for (const auto& [column, value] : source.transition)
    target.transition[column] += factor * value;
  for (const auto& [column, value] : source.detection)
    target.detection[column] += factor * value;
  for (const auto& [column, value] : source.losses)
    target.losses[column] += factor * value;
}

void scale_row(RowAccumulation& row, double factor) {
  for (auto& [column, value] : row.transition) {
    (void)column;
    value *= factor;
  }
  for (auto& [column, value] : row.detection) {
    (void)column;
    value *= factor;
  }
  for (auto& [column, value] : row.losses) {
    (void)column;
    value *= factor;
  }
}

double map_l1_difference(const std::map<std::uint32_t, double>& first,
                         const std::map<std::uint32_t, double>& second) {
  auto a = first.begin();
  auto b = second.begin();
  double result = 0.0;
  while (a != first.end() || b != second.end()) {
    if (b == second.end() ||
        (a != first.end() && a->first < b->first)) {
      result += std::abs(a->second);
      ++a;
    } else if (a == first.end() || b->first < a->first) {
      result += std::abs(b->second);
      ++b;
    } else {
      result += std::abs(a->second - b->second);
      ++a;
      ++b;
    }
  }
  return result;
}

double row_l1_difference(const RowAccumulation& first,
                         const RowAccumulation& second) {
  return map_l1_difference(first.transition, second.transition) +
         map_l1_difference(first.detection, second.detection) +
         map_l1_difference(first.losses, second.losses);
}

Vec3 shape_factor_reference_axis(const Vec3& direction) {
  const Vec3 trial =
      std::abs(direction.z) < 0.9 ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
  return normalized(cross(trial, direction));
}

struct ShapeFactorTriangle {
  Vec3 a;
  Vec3 b;
  Vec3 c;
  std::uint32_t primitive{};
  std::uint32_t depth{};
  // Flags correspond to edges (a,b), (b,c), and (c,a).
  std::array<bool, 3> feature_edges{};
};

double exact_triangle_solid_angle(const Vec3& point,
                                  const ShapeFactorTriangle& triangle) {
  const Vec3 a = subtract(triangle.a, point);
  const Vec3 b = subtract(triangle.b, point);
  const Vec3 c = subtract(triangle.c, point);
  const double la = norm(a);
  const double lb = norm(b);
  const double lc = norm(c);
  if (!(la > 0.0) || !(lb > 0.0) || !(lc > 0.0))
    throw std::runtime_error(
        "shape-factor source lies on a boundary triangle");
  const double numerator = std::abs(dot(a, cross(b, c)));
  const double denominator =
      la * lb * lc + dot(a, b) * lc + dot(b, c) * la +
      dot(c, a) * lb;
  return 2.0 * std::atan2(numerator, denominator);
}

std::array<ShapeFactorTriangle, 4> subdivide_shape_factor_triangle(
    const ShapeFactorTriangle& triangle) {
  const Vec3 ab = scale(add(triangle.a, triangle.b), 0.5);
  const Vec3 bc = scale(add(triangle.b, triangle.c), 0.5);
  const Vec3 ca = scale(add(triangle.c, triangle.a), 0.5);
  const auto next_depth = triangle.depth + 1;
  return {
      ShapeFactorTriangle{
          triangle.a, ab, ca, triangle.primitive, next_depth,
          {triangle.feature_edges[0], false, triangle.feature_edges[2]}},
      ShapeFactorTriangle{
          ab, triangle.b, bc, triangle.primitive, next_depth,
          {triangle.feature_edges[0], triangle.feature_edges[1], false}},
      ShapeFactorTriangle{
          ca, bc, triangle.c, triangle.primitive, next_depth,
          {false, triangle.feature_edges[1], triangle.feature_edges[2]}},
      ShapeFactorTriangle{ab, bc, ca, triangle.primitive, next_depth,
                          {false, false, false}}};
}

using ShapeFactorFeatureEdges = std::vector<std::array<bool, 3>>;

ShapeFactorFeatureEdges build_shape_factor_feature_edges(
    const Scene& scene, double dihedral_degrees) {
  struct EdgeOwner {
    std::uint32_t primitive{};
    std::uint32_t local_edge{};
  };
  const auto edge_key = [](std::uint32_t first, std::uint32_t second) {
    const auto low = std::min(first, second);
    const auto high = std::max(first, second);
    return (static_cast<std::uint64_t>(low) << 32u) | high;
  };
  const auto domain_pair = [&](std::uint32_t primitive) {
    auto first = scene.mesh.minus_domain_id.at(primitive);
    auto second = scene.mesh.plus_domain_id.at(primitive);
    if (second < first) std::swap(first, second);
    return std::pair{first, second};
  };
  ShapeFactorFeatureEdges result(scene.mesh.triangles.size());
  std::unordered_map<std::uint64_t, EdgeOwner> owners;
  owners.reserve(scene.mesh.triangles.size() * 2);
  std::vector<Vec3> normals(scene.mesh.triangles.size());
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive)
    normals[primitive] = triangle_normal(scene, primitive);
  const double minimum_normal_cosine =
      std::cos(dihedral_degrees * pi / 180.0);
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    const auto triangle = scene.mesh.triangles.at(primitive);
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> edges{{
        {triangle[0], triangle[1]},
        {triangle[1], triangle[2]},
        {triangle[2], triangle[0]},
    }};
    for (std::uint32_t local_edge = 0; local_edge < edges.size();
         ++local_edge) {
      const auto key =
          edge_key(edges[local_edge].first, edges[local_edge].second);
      const auto [found, inserted] =
          owners.emplace(key, EdgeOwner{primitive, local_edge});
      if (inserted) continue;
      const auto other = found->second;
      const double normal_cosine =
          std::abs(dot(normals[primitive], normals[other.primitive]));
      const bool feature =
          scene.mesh.surface_id.at(primitive) !=
              scene.mesh.surface_id.at(other.primitive) ||
          domain_pair(primitive) != domain_pair(other.primitive) ||
          normal_cosine < minimum_normal_cosine;
      if (feature) {
        result[primitive][local_edge] = true;
        result[other.primitive][other.local_edge] = true;
      }
    }
  }
  return result;
}

struct ShapeFactorTraceContext {
  const Scene& scene;
  const Geometry& geometry;
  const std::unordered_map<std::uint64_t, std::uint32_t>& primitive_to_state;
  const std::unordered_map<std::int32_t, std::uint32_t>& channel_to_column;
  const CustomRuntimeMap& custom_runtimes;
  const SourcePoint& source;
  const ShapeFactorOptions& options;
};

RowAccumulation evaluate_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle) {
  RowAccumulation result;
  const Vec3 center =
      scale(add(add(triangle.a, triangle.b), triangle.c), 1.0 / 3.0);
  const Vec3 displacement = subtract(center, context.source.position);
  const double distance = norm(displacement);
  if (!(distance > 0.0))
    throw std::runtime_error(
        "shape-factor source coincides with a triangle centroid");
  const Vec3 direction = scale(displacement, 1.0 / distance);
  const Hit first = context.geometry.intersect(
      {context.source.position, direction}, context.source.domain);
  // A non-convex boundary can contain triangles hidden behind the first
  // domain boundary.  They do not own any first-flight solid angle.  Mixed
  // visibility is resolved by the child comparison below.
  const bool exact_triangle =
      first.valid && first.kind == GeometryPrimitiveKind::triangle &&
      first.primitive_id == triangle.primitive;
  const bool analytic_replacement =
      first.valid &&
      !context.scene.mesh.triangle_transport.empty() &&
      !context.scene.mesh.triangle_transport.at(triangle.primitive) &&
      !context.scene.mesh.triangle_source_quadrature.empty() &&
      context.scene.mesh.triangle_source_quadrature.at(
          triangle.primitive) &&
      first.kind != GeometryPrimitiveKind::triangle &&
      first.surface_id ==
          context.scene.mesh.surface_id.at(triangle.primitive) &&
      (first.minus_domain_id == context.source.domain ||
       first.plus_domain_id == context.source.domain);
  if (!exact_triangle && !analytic_replacement)
    return result;
  const double solid_angle =
      exact_triangle_solid_angle(context.source.position, triangle);
  if (!(solid_angle > 0.0)) return result;
  const double weight =
      context.source.stokes.i * solid_angle / (4.0 * pi);
  trace_branch(
      context.scene, context.geometry, context.primitive_to_state,
      context.channel_to_column, context.custom_runtimes,
      {{context.source.position, direction},
       {weight, 0.0, 0.0, 0.0},
       shape_factor_reference_axis(direction), context.source.domain, 0,
       first},
      result);
  return result;
}

double projected_aperture_coverage(
    const ShapeFactorTraceContext& context,
    const AnalyticSurfaceElement& element, double unmasked_weight) {
  const auto missing = std::numeric_limits<std::uint32_t>::max();
  if (element.projected_aperture_primitive_index == missing &&
      element.projected_aperture_hole_index == missing)
    return 1.0;
  if (element.projected_aperture_primitive_index == missing ||
      element.projected_aperture_hole_index == missing)
    throw std::runtime_error(
        "shape-factor element has an incomplete projected aperture");
  if (element.coordinates !=
      AnalyticSurfaceCoordinates::annulus_r2_phi)
    throw std::runtime_error(
        "projected aperture currently requires polar area cells");
  const auto& receiver = context.scene.mesh.analytic_primitives.at(
      element.primitive_index);
  const auto& aperture = context.scene.mesh.analytic_primitives.at(
      element.projected_aperture_primitive_index);
  if (aperture.kind != GeometryPrimitiveKind::perforated_disk)
    throw std::runtime_error(
        "projected aperture primitive must be a perforated disk");
  if (element.projected_aperture_hole_index >= aperture.holes.size())
    throw std::runtime_error(
        "projected aperture hole index is out of range");
  const auto& hole =
      aperture.holes.at(element.projected_aperture_hole_index);
  const Vec3 source = context.source.position;
  const double denominator =
      dot(subtract(receiver.center_mm, source), aperture.axis_z);
  if (std::abs(denominator) <=
      64.0 * std::numeric_limits<double>::epsilon())
    throw std::runtime_error(
        "receiver and aperture projection are degenerate");
  const double fraction =
      dot(subtract(aperture.center_mm, source), aperture.axis_z) /
      denominator;
  const Vec3 hole_center =
      add(aperture.center_mm,
          add(scale(aperture.axis_x, hole.center_uv_mm.x),
              scale(aperture.axis_y, hole.center_uv_mm.y)));
  const auto aperture_coordinates = [&](const Vec3& point) {
    const Vec3 projected =
        add(source, scale(subtract(point, source), fraction));
    const Vec3 relative = subtract(projected, hole_center);
    return Vec2{dot(relative, aperture.axis_x),
                dot(relative, aperture.axis_y)};
  };
  const Vec2 center = aperture_coordinates(element.center_mm);
  const double center_distance =
      std::hypot(center.x, center.y);
  const double radial_min =
      std::sqrt(std::max(0.0, element.bounds[0]));
  const double radial_max =
      std::sqrt(std::max(0.0, element.bounds[1]));
  const double phi_min = element.bounds[2];
  const double phi_max = element.bounds[3];
  const Vec3 receiver_relative =
      subtract(element.center_mm, receiver.center_mm);
  const Vec2 sample{
      dot(receiver_relative, receiver.axis_x),
      dot(receiver_relative, receiver.axis_y)};
  double cell_radius = 0.0;
  for (const double radius : {radial_min, radial_max})
    for (const double phi : {phi_min, phi_max}) {
      const Vec2 corner{radius * std::cos(phi),
                        radius * std::sin(phi)};
      cell_radius =
          std::max(cell_radius,
                   std::hypot(corner.x - sample.x,
                              corner.y - sample.y));
    }
  const double uncertainty = std::abs(fraction) * cell_radius;
  if (center_distance + uncertainty <= hole.radius_mm) return 1.0;
  if (center_distance - uncertainty >= hole.radius_mm) return 0.0;
  const double center_decision =
      center_distance <= hole.radius_mm ? 1.0 : 0.0;
  if (unmasked_weight <
      context.options.aperture_edge_weight_threshold)
    return center_decision;

  // Parameterize the finite receiver cell around the receiver primitive,
  // then project each radial direction into the aperture plane.  This is the
  // finite-area circular clipping rule, generalized to arbitrary
  // coplanar local frames.
  const Vec2 projected_receiver_center =
      aperture_coordinates(receiver.center_mm);
  double radial_area_integral = 0.0;
  for (const auto& [unit_node, unit_weight] :
       gauss_legendre_unit_interval(
           context.options.aperture_edge_phi_order)) {
    const double phi =
        phi_min + (phi_max - phi_min) * unit_node;
    const Vec3 radial_direction =
        add(scale(receiver.axis_x, std::cos(phi)),
            scale(receiver.axis_y, std::sin(phi)));
    const Vec2 projected_direction{
        fraction * dot(radial_direction, aperture.axis_x),
        fraction * dot(radial_direction, aperture.axis_y)};
    const double quadratic =
        projected_direction.x * projected_direction.x +
        projected_direction.y * projected_direction.y;
    const double linear =
        2.0 * (projected_receiver_center.x *
                   projected_direction.x +
               projected_receiver_center.y *
                   projected_direction.y);
    const double constant =
        projected_receiver_center.x * projected_receiver_center.x +
        projected_receiver_center.y * projected_receiver_center.y -
        hole.radius_mm * hole.radius_mm;
    const double discriminant =
        linear * linear - 4.0 * quadratic * constant;
    if (!(quadratic > 0.0) || !(discriminant > 0.0)) continue;
    const double root = std::sqrt(discriminant);
    const double lower_root =
        (-linear - root) / (2.0 * quadratic);
    const double upper_root =
        (-linear + root) / (2.0 * quadratic);
    const double lower = std::max({radial_min, lower_root, 0.0});
    const double upper = std::min(radial_max, upper_root);
    if (upper > lower)
      radial_area_integral +=
          unit_weight * (upper * upper - lower * lower);
  }
  const double covered_area =
      0.5 * (phi_max - phi_min) * radial_area_integral;
  const double cell_area =
      0.5 * (radial_max * radial_max -
             radial_min * radial_min) *
      (phi_max - phi_min);
  if (!(cell_area > 0.0))
    throw std::runtime_error(
        "projected aperture cell has zero area");
  return std::clamp(covered_area / cell_area, 0.0, 1.0);
}

RowAccumulation evaluate_shape_factor_analytic_element(
    const ShapeFactorTraceContext& context, std::uint32_t element_index) {
  RowAccumulation result;
  const auto& element =
      context.scene.mesh.analytic_surface_elements.at(element_index);
  const Vec3 displacement =
      subtract(element.center_mm, context.source.position);
  const double distance = norm(displacement);
  if (!(distance > 0.0))
    throw std::runtime_error(
        "shape-factor source coincides with an analytic surface element");
  const Vec3 direction = scale(displacement, 1.0 / distance);
  const double projected_cosine =
      std::max(0.0, dot(direction, element.normal));
  if (!(projected_cosine > 0.0)) return result;
  const auto expected_key =
      analytic_surface_element_geometry_key(element_index);
  Hit first = context.geometry.intersect(
      {context.source.position, direction}, context.source.domain);
  const double unmasked_weight =
      context.source.stokes.i * projected_cosine * element.area_mm2 /
      (4.0 * pi * distance * distance);
  const double coverage = projected_aperture_coverage(
      context, element, unmasked_weight);
  if (!(coverage > 0.0)) return result;
  if (element.projected_aperture_primitive_index !=
      std::numeric_limits<std::uint32_t>::max()) {
    // The aperture can hide the center ray while exposing a finite fraction
    // of the receiver cell.  Once the exact covered fraction is known, route
    // that fraction to the receiver surface rather than the blocking plane.
    const auto& primitive =
        context.scene.mesh.analytic_primitives.at(
            element.primitive_index);
    first = {
        true,
        primitive.kind,
        expected_key,
        element.primitive_index,
        primitive.surface_id,
        distance,
        element.normal,
        primitive.minus_domain_id,
        primitive.plus_domain_id,
        primitive.channel_id,
        element.surface_basis_id,
        element.surface_element,
        {1.0, 0.0, 0.0}};
  } else if (!first.valid || first.geometry_key != expected_key) {
    return result;
  }
  const double weight = unmasked_weight * coverage;
  if (!(weight > 0.0) || !std::isfinite(weight))
    throw std::runtime_error(
        "analytic shape-factor element produced invalid weight");
  trace_branch(
      context.scene, context.geometry, context.primitive_to_state,
      context.channel_to_column, context.custom_runtimes,
      {{context.source.position, direction},
       {weight, 0.0, 0.0, 0.0},
       shape_factor_reference_axis(direction),
       context.source.domain,
       0,
       first},
      result);
  return result;
}

struct ShapeFactorAssessment {
  std::array<ShapeFactorTriangle, 4> children;
  std::array<RowAccumulation, 4> child_evaluations;
  RowAccumulation child_sum;
  double error{};
};

ShapeFactorAssessment assess_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle,
    std::optional<RowAccumulation> parent_evaluation = std::nullopt) {
  RowAccumulation parent =
      parent_evaluation
          ? std::move(*parent_evaluation)
          : evaluate_shape_factor_triangle(context, triangle);
  ShapeFactorAssessment result;
  result.children = subdivide_shape_factor_triangle(triangle);
  for (std::size_t child = 0; child < result.children.size(); ++child) {
    result.child_evaluations[child] =
        evaluate_shape_factor_triangle(context, result.children[child]);
    add_row(result.child_sum, result.child_evaluations[child]);
  }
  result.error = row_l1_difference(parent, result.child_sum);
  return result;
}

struct ShapeFactorIntegral {
  RowAccumulation row;
  double estimated_l1_error{};
};

ShapeFactorIntegral integrate_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle, double error_budget,
    double numerical_error_floor,
    std::optional<RowAccumulation> parent_evaluation = std::nullopt) {
  RowAccumulation parent =
      parent_evaluation
          ? std::move(*parent_evaluation)
          : evaluate_shape_factor_triangle(context, triangle);
  const double solid_angle_fraction =
      context.source.stokes.i *
      exact_triangle_solid_angle(context.source.position, triangle) /
      (4.0 * pi);
  const bool touches_feature =
      std::any_of(triangle.feature_edges.begin(),
                  triangle.feature_edges.end(),
                  [](bool value) { return value; });
  const double refinement_threshold =
      touches_feature
          ? context.options.minimum_feature_solid_angle_fraction
          : context.options.minimum_refinement_solid_angle_fraction;
  if (solid_angle_fraction < refinement_threshold)
    return {std::move(parent), 0.0};

  ShapeFactorAssessment assessment =
      assess_shape_factor_triangle(context, triangle, std::move(parent));
  const double effective_budget =
      std::max(error_budget, numerical_error_floor);
  if (assessment.error <= effective_budget)
    return {std::move(assessment.child_sum), assessment.error};
  if (triangle.depth >= context.options.maximum_subdivision_depth) {
    // Deterministic hard cap: keep the more accurate child sum and expose the
    // unresolved parent/child difference through the source integration audit.
    return {std::move(assessment.child_sum), assessment.error};
  }

  ShapeFactorIntegral result;
  const double child_weight_sum = row_total(assessment.child_sum);
  for (std::size_t child = 0; child < assessment.children.size(); ++child) {
    const double fraction =
        child_weight_sum > 0.0
            ? row_total(assessment.child_evaluations[child]) /
                  child_weight_sum
            : 0.25;
    auto refined = integrate_shape_factor_triangle(
        context, assessment.children[child], error_budget * fraction,
        numerical_error_floor,
        std::move(assessment.child_evaluations[child]));
    add_row(result.row, refined.row);
    result.estimated_l1_error += refined.estimated_l1_error;
  }
  return result;
}

ShapeFactorIntegral trace_shape_factor_point(
    const Scene& scene, const Geometry& geometry,
    const std::unordered_map<std::uint64_t, std::uint32_t>&
        primitive_to_state,
    const std::unordered_map<std::int32_t, std::uint32_t>&
        channel_to_column,
    const CustomRuntimeMap& custom_runtimes,
    const ShapeFactorFeatureEdges& feature_edges,
    const SourcePoint& source, const ShapeFactorOptions& options) {
  if (std::abs(source.stokes.q) > scene.numerics.energy_tolerance ||
      std::abs(source.stokes.u) > scene.numerics.energy_tolerance ||
      std::abs(source.stokes.v) > scene.numerics.energy_tolerance)
    throw std::runtime_error(
        "isotropic shape-factor sources currently require unpolarized "
        "Stokes input");
  if (scene.media.find(source.domain) == scene.media.end())
    throw std::runtime_error("shape-factor source domain is not declared");

  ShapeFactorTraceContext context{
      scene, geometry, primitive_to_state, channel_to_column,
      custom_runtimes, source, options};
  std::vector<std::uint32_t> boundary_primitives;
  boundary_primitives.reserve(scene.mesh.triangles.size());
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    const auto minus = scene.mesh.minus_domain_id.at(primitive);
    const auto plus = scene.mesh.plus_domain_id.at(primitive);
    if (source.domain != minus && source.domain != plus) continue;
    const bool source_quadrature =
        scene.mesh.triangle_source_quadrature.empty()
            ? (scene.mesh.triangle_transport.empty() ||
               scene.mesh.triangle_transport.at(primitive) != 0)
            : scene.mesh.triangle_source_quadrature.at(primitive) != 0;
    if (!source_quadrature)
      continue;
    const auto indices = scene.mesh.triangles.at(primitive);
    const Vec3 center =
        scale(add(add(scene.mesh.vertices.at(indices[0]),
                      scene.mesh.vertices.at(indices[1])),
                  scene.mesh.vertices.at(indices[2])),
              1.0 / 3.0);
    const Vec3 geometric_normal = triangle_normal(scene, primitive);
    const Vec3 outward =
        source.domain == minus ? geometric_normal
                               : scale(geometric_normal, -1.0);
    if (dot(subtract(center, source.position), outward) > 0.0)
      boundary_primitives.push_back(primitive);
  }
  std::vector<std::uint32_t> analytic_elements;
  analytic_elements.reserve(scene.mesh.analytic_surface_elements.size());
  for (std::uint32_t index = 0;
       index < scene.mesh.analytic_surface_elements.size(); ++index) {
    const auto& element =
        scene.mesh.analytic_surface_elements.at(index);
    if (!element.source_quadrature) continue;
    const auto& primitive =
        scene.mesh.analytic_primitives.at(element.primitive_index);
    if (source.domain != primitive.minus_domain_id &&
        source.domain != primitive.plus_domain_id)
      continue;
    if (dot(subtract(element.center_mm, source.position),
            element.normal) > 0.0)
      analytic_elements.push_back(index);
  }
  const std::size_t boundary_count =
      boundary_primitives.size() + analytic_elements.size();
  if (boundary_count == 0)
    throw std::runtime_error(
        "shape-factor source has no outward-facing domain boundary");
  const double global_error_budget =
      options.relative_tolerance * source.stokes.i;
  const double primitive_error_budget =
      global_error_budget / boundary_count;
  const double primitive_numerical_error_floor =
      scene.numerics.energy_tolerance * source.stokes.i /
      boundary_count;

  RowAccumulation result;
  double estimated_l1_error = 0.0;
  std::exception_ptr error;
#pragma omp parallel
  {
    RowAccumulation local;
    double local_estimated_l1_error = 0.0;
    std::exception_ptr local_error;
#pragma omp for schedule(dynamic, 1)
    for (std::int64_t boundary_index = 0;
         boundary_index <
         static_cast<std::int64_t>(boundary_count);
         ++boundary_index) {
      if (local_error) continue;
      try {
        if (boundary_index >=
            static_cast<std::int64_t>(boundary_primitives.size())) {
          const auto analytic_index =
              analytic_elements[boundary_index -
                                boundary_primitives.size()];
          add_row(local, evaluate_shape_factor_analytic_element(
                             context, analytic_index));
          continue;
        }
        const auto primitive = boundary_primitives[boundary_index];
        const auto indices = scene.mesh.triangles.at(primitive);
        ShapeFactorTriangle triangle{
            scene.mesh.vertices.at(indices[0]),
            scene.mesh.vertices.at(indices[1]),
            scene.mesh.vertices.at(indices[2]), primitive, 0,
            feature_edges.at(primitive)};
        auto integrated = integrate_shape_factor_triangle(
            context, triangle, primitive_error_budget,
            primitive_numerical_error_floor);
        add_row(local, integrated.row);
        local_estimated_l1_error += integrated.estimated_l1_error;
      } catch (...) {
        local_error = std::current_exception();
      }
    }
#pragma omp critical(oos_shape_factor_merge)
    {
      add_row(result, local);
      estimated_l1_error += local_estimated_l1_error;
      if (local_error && !error) error = local_error;
    }
  }
  if (error) std::rethrow_exception(error);
  const double accounted = row_total(result);
  if (!(accounted > 0.0) || !std::isfinite(accounted))
    throw std::runtime_error(
        "shape-factor source produced no finite boundary weight");
  // A hard cap can stop while a non-convex boundary triangle is only partly
  // visible. Its centroid decision then temporarily under- or over-counts a
  // small solid-angle region. Deterministic source quadratures are normalized
  // by contract; include the normalization correction in the L1 error audit.
  estimated_l1_error += std::abs(accounted - source.stokes.i);
  scale_row(result, source.stokes.i / accounted);
  return {std::move(result), estimated_l1_error};
}

CsrMatrix maps_to_csr(
    const std::vector<std::map<std::uint32_t, double>>& rows,
    std::uint64_t columns) {
  CsrMatrix matrix;
  matrix.rows = rows.size();
  matrix.cols = columns;
  matrix.indptr.push_back(0);
  for (const auto& row : rows) {
    for (const auto& [column, value] : row) {
      if (value == 0.0) continue;
      matrix.indices.push_back(column);
      matrix.data.push_back(value);
    }
    matrix.indptr.push_back(matrix.data.size());
  }
  return matrix;
}

CsrMatrix payload_csr(const PluginBuildResult& payload,
                      const std::string& root) {
  const auto& shape = payload.u64.at(root + "/shape").values;
  const auto& indptr = payload.u64.at(root + "/indptr").values;
  const auto& index_values = payload.u64.at(root + "/indices").values;
  CsrMatrix result;
  if (shape.size() != 2) throw std::runtime_error("plugin CSR shape is invalid");
  result.rows = shape[0];
  result.cols = shape[1];
  result.indptr = indptr;
  result.indices.reserve(index_values.size());
  for (const auto value : index_values) {
    if (value > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("plugin CSR index exceeds uint32");
    result.indices.push_back(static_cast<std::uint32_t>(value));
  }
  result.data = payload.f64.at(root + "/data").values;
  return result;
}

void apply_provenance(OperatorSet& result, const Scene& scene) {
  result.energy_eV = scene.energy_eV;
  result.scene_sha256 =
      scene.source_path.empty() ? "in-memory" : sha256_file(scene.source_path);
  result.geometry_sha256 = scene.geometry_path.empty()
                               ? "in-memory"
                               : sha256_file(scene.geometry_path);
  result.surface_basis_sha256 =
      scene.surface_basis_path.empty()
          ? "embedded"
          : sha256_file(scene.surface_basis_path);
#ifdef OOS_DEPS_LOCK_SHA256
  result.dependency_lock_sha256 = OOS_DEPS_LOCK_SHA256;
#endif
#ifdef OOS_GIT_COMMIT
  result.code_commit = OOS_GIT_COMMIT;
#endif
  std::string cache_material =
      result.scene_sha256 + result.geometry_sha256 +
      result.surface_basis_sha256 +
      result.dependency_lock_sha256 + result.code_commit +
      std::to_string(result.energy_eV);
  for (const auto& [id, surface] : scene.surfaces) {
    (void)id;
    if (surface.kind == SurfaceKind::custom_local ||
        surface.kind == SurfaceKind::custom_nonlocal) {
      cache_material += sha256_file(surface.plugin_path);
      if (!surface.plugin_config_json.empty()) {
        const auto config =
            nlohmann::json::parse(surface.plugin_config_json);
        for (const auto* key :
             {"precomputed_operator_hdf5", "precomputed_block_hdf5",
              "factorized_block_hdf5"}) {
          if (!config.contains(key)) continue;
          const auto path =
              std::filesystem::path(config.at(key).get<std::string>());
          if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error(
                std::string("custom operator input is not a regular file: ") +
                path.string());
          cache_material += key;
          cache_material += sha256_file(path);
        }
      }
    }
  }
  result.cache_key_sha256 = sha256_string(cache_material);
}

std::vector<std::string> payload_loss_names(
    const PluginBuildResult& payload, std::uint64_t count) {
  const auto metadata = nlohmann::json::parse(payload.metadata_json);
  if (!metadata.contains("loss_names"))
    throw std::runtime_error("nonlocal plugin metadata lacks loss_names");
  const auto names =
      metadata.at("loss_names").get<std::vector<std::string>>();
  if (names.size() != count)
    throw std::runtime_error("nonlocal plugin loss_names shape mismatch");
  return names;
}

struct NonlocalEgress {
  std::uint64_t surface_element{};
  std::array<double, 3> barycentric{};
  std::uint64_t side{};
  Vec3 direction_local{};
  Stokes stokes{1.0, 0.0, 0.0, 0.0};
  Vec3 reference_axis_local{1.0, 0.0, 0.0};
};

std::vector<NonlocalEgress> payload_egress(
    const PluginBuildResult& payload) {
  const auto& elements =
      payload.u64.at("/nonlocal/egress/surface_element");
  const auto& barycentric =
      payload.f64.at("/nonlocal/egress/barycentric");
  const auto& sides = payload.u64.at("/nonlocal/egress/side");
  const auto& directions =
      payload.f64.at("/nonlocal/egress/direction_local");
  const auto stokes =
      payload.f64.find("/nonlocal/egress/stokes");
  const auto reference_axes =
      payload.f64.find("/nonlocal/egress/reference_axis_local");
  if (elements.shape.size() != 1 || sides.shape != elements.shape ||
      barycentric.shape !=
          std::vector<std::uint64_t>{elements.shape[0], 3} ||
      directions.shape !=
          std::vector<std::uint64_t>{elements.shape[0], 3} ||
      (stokes != payload.f64.end() &&
       stokes->second.shape !=
           std::vector<std::uint64_t>{elements.shape[0], 4}) ||
      (reference_axes != payload.f64.end() &&
       reference_axes->second.shape !=
           std::vector<std::uint64_t>{elements.shape[0], 3}))
    throw std::runtime_error("nonlocal egress dataset shapes disagree");
  std::vector<NonlocalEgress> result(elements.shape[0]);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i].surface_element = elements.values.at(i);
    result[i].barycentric = {barycentric.values.at(3 * i),
                             barycentric.values.at(3 * i + 1),
                             barycentric.values.at(3 * i + 2)};
    result[i].side = sides.values.at(i);
    result[i].direction_local = {directions.values.at(3 * i),
                                 directions.values.at(3 * i + 1),
                                 directions.values.at(3 * i + 2)};
    if (stokes != payload.f64.end())
      result[i].stokes = {stokes->second.values.at(4 * i),
                          stokes->second.values.at(4 * i + 1),
                          stokes->second.values.at(4 * i + 2),
                          stokes->second.values.at(4 * i + 3)};
    if (reference_axes != payload.f64.end())
      result[i].reference_axis_local = {
          reference_axes->second.values.at(3 * i),
          reference_axes->second.values.at(3 * i + 1),
          reference_axes->second.values.at(3 * i + 2)};
    const double barycentric_sum =
        std::accumulate(result[i].barycentric.begin(),
                        result[i].barycentric.end(), 0.0);
    for (const auto value : result[i].barycentric)
      if (!std::isfinite(value) || value < 0.0)
        throw std::runtime_error(
            "nonlocal egress barycentric coordinate is invalid");
    if (std::abs(barycentric_sum - 1.0) > 1.0e-12)
      throw std::runtime_error(
          "nonlocal egress barycentric coordinates do not close");
    if (result[i].side > 1)
      throw std::runtime_error("nonlocal egress side must be zero or one");
    if (!std::isfinite(result[i].direction_local.x) ||
        !std::isfinite(result[i].direction_local.y) ||
        !std::isfinite(result[i].direction_local.z) ||
        result[i].direction_local.z <= 0.0)
      throw std::runtime_error(
          "nonlocal egress direction must point into its declared side");
    result[i].direction_local = normalized(result[i].direction_local);
    const auto& polarization = result[i].stokes;
    if (!std::isfinite(polarization.i) ||
        !std::isfinite(polarization.q) ||
        !std::isfinite(polarization.u) ||
        !std::isfinite(polarization.v) ||
        std::abs(polarization.i - 1.0) > 1.0e-12 ||
        polarization.q * polarization.q +
                polarization.u * polarization.u +
                polarization.v * polarization.v >
            1.0 + 1.0e-12)
      throw std::runtime_error(
          "nonlocal egress Stokes vector must have unit intensity and "
          "physical polarization");
    if (!std::isfinite(result[i].reference_axis_local.x) ||
        !std::isfinite(result[i].reference_axis_local.y) ||
        !std::isfinite(result[i].reference_axis_local.z))
      throw std::runtime_error(
          "nonlocal egress reference axis is invalid");
  }
  return result;
}

std::vector<std::uint32_t> surface_primitives(const Scene& scene,
                                              std::uint32_t surface_id) {
  std::vector<std::uint32_t> result;
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.surface_id.size(); ++primitive)
    if (scene.mesh.surface_id[primitive] == surface_id &&
        (scene.mesh.triangle_transport.empty() ||
         scene.mesh.triangle_transport.at(primitive) != 0))
      result.push_back(primitive);
  return result;
}

WeightedRay egress_ray(const Scene& scene, std::uint32_t surface_id,
                       const NonlocalEgress& egress, double nudge,
                       double weight = 1.0) {
  const AnalyticSurfaceElement* analytic_element = nullptr;
  const AnalyticPrimitive* analytic_primitive = nullptr;
  for (const auto& element : scene.mesh.analytic_surface_elements) {
    const auto& primitive =
        scene.mesh.analytic_primitives.at(element.primitive_index);
    if (primitive.surface_id != surface_id ||
        element.surface_element != egress.surface_element)
      continue;
    if (analytic_element != nullptr)
      throw std::runtime_error(
          "nonlocal egress surface element is ambiguous");
    analytic_element = &element;
    analytic_primitive = &primitive;
  }
  if (analytic_element != nullptr) {
    const Vec3 point = analytic_element->center_mm;
    const Vec3 geometric_normal = normalized(analytic_element->normal);
    const Vec3 side_normal =
        egress.side == 1 ? geometric_normal
                         : scale(geometric_normal, -1.0);
    Vec3 tangent = analytic_primitive->axis_x;
    if (analytic_element->coordinates ==
        AnalyticSurfaceCoordinates::cylinder_phi_z) {
      const double phi =
          0.5 * (analytic_element->bounds[0] +
                 analytic_element->bounds[1]);
      tangent = normalized(
          add(scale(analytic_primitive->axis_x, -std::sin(phi)),
              scale(analytic_primitive->axis_y, std::cos(phi))));
    }
    tangent = normalized(
        subtract(tangent, scale(side_normal, dot(tangent, side_normal))));
    const Vec3 bitangent = cross(side_normal, tangent);
    const Vec3 direction = normalized(
        add(scale(tangent, egress.direction_local.x),
            add(scale(bitangent, egress.direction_local.y),
                scale(side_normal, egress.direction_local.z))));
    Vec3 reference_axis =
        add(scale(tangent, egress.reference_axis_local.x),
            add(scale(bitangent, egress.reference_axis_local.y),
                scale(side_normal, egress.reference_axis_local.z)));
    reference_axis =
        subtract(reference_axis,
                 scale(direction, dot(reference_axis, direction)));
    if (norm(reference_axis) < 1.0e-14)
      reference_axis =
          subtract(tangent, scale(direction, dot(tangent, direction)));
    if (norm(reference_axis) < 1.0e-14)
      reference_axis =
          subtract(bitangent, scale(direction, dot(bitangent, direction)));
    reference_axis = normalized(reference_axis);
    const auto domain =
        egress.side == 1 ? analytic_primitive->plus_domain_id
                         : analytic_primitive->minus_domain_id;
    if (domain < 0)
      throw std::runtime_error(
          "nonlocal analytic egress points outside all declared media");
    return {{add(point, scale(side_normal, nudge)), direction},
            {weight, weight * egress.stokes.q, weight * egress.stokes.u,
             weight * egress.stokes.v},
            reference_axis, domain, 0, std::nullopt};
  }
  const auto primitives = surface_primitives(scene, surface_id);
  if (egress.surface_element >= primitives.size())
    throw std::runtime_error(
        "nonlocal egress surface element lies outside its own surface");
  const auto primitive = primitives.at(egress.surface_element);
  const auto triangle = scene.mesh.triangles.at(primitive);
  Vec3 point{};
  for (int vertex = 0; vertex < 3; ++vertex)
    point = add(point,
                scale(scene.mesh.vertices.at(triangle[vertex]),
                      egress.barycentric[vertex]));
  const Vec3 geometric_normal = triangle_normal(scene, primitive);
  const Vec3 side_normal =
      egress.side == 1 ? geometric_normal : scale(geometric_normal, -1.0);
  const Vec3 tangent = normalized(
      subtract(scene.mesh.vertices.at(triangle[1]),
               scene.mesh.vertices.at(triangle[0])));
  const Vec3 bitangent = cross(side_normal, tangent);
  const Vec3 direction = normalized(
      add(scale(tangent, egress.direction_local.x),
          add(scale(bitangent, egress.direction_local.y),
              scale(side_normal, egress.direction_local.z))));
  Vec3 reference_axis =
      add(scale(tangent, egress.reference_axis_local.x),
          add(scale(bitangent, egress.reference_axis_local.y),
              scale(side_normal, egress.reference_axis_local.z)));
  reference_axis =
      subtract(reference_axis,
               scale(direction, dot(reference_axis, direction)));
  if (norm(reference_axis) < 1.0e-14)
    reference_axis =
        subtract(tangent, scale(direction, dot(tangent, direction)));
  if (norm(reference_axis) < 1.0e-14)
    reference_axis =
        subtract(bitangent, scale(direction, dot(bitangent, direction)));
  reference_axis = normalized(reference_axis);
  const auto domain =
      egress.side == 1 ? scene.mesh.plus_domain_id.at(primitive)
                       : scene.mesh.minus_domain_id.at(primitive);
  if (domain < 0)
    throw std::runtime_error(
        "nonlocal egress ray points outside all declared media");
  return {{add(point, scale(side_normal, nudge)),
           direction},
          {weight, weight * egress.stokes.q, weight * egress.stokes.u,
           weight * egress.stokes.v},
          reference_axis, domain, 0, std::nullopt};
}

void validate_intrinsic_nonlocal_payload(const PluginBuildResult& payload,
                                         double tolerance) {
  const std::array<std::string, 5> forbidden{
      "/nonlocal/detection", "/nonlocal/channel_id",
      "/nonlocal/to_local", "/nonlocal/to_local_primitive_id", "/operators"};
  const auto reject_forbidden = [&forbidden](const auto& datasets) {
    for (const auto& [path, dataset] : datasets) {
      (void)dataset;
      for (const auto& prefix : forbidden)
        if (path == prefix || path.rfind(prefix + "/", 0) == 0)
          throw std::runtime_error(
              "nonlocal surface payload contains geometry-coupled field " +
              path);
    }
  };
  reject_forbidden(payload.f64);
  reject_forbidden(payload.u64);
  const auto metadata = nlohmann::json::parse(payload.metadata_json);
  if (metadata.value("execution", std::string{}) == "function") {
    const auto egress = payload_egress(payload);
    if (!metadata.contains("state_count") ||
        !metadata.contains("loss_names") ||
        metadata.at("state_count").get<std::uint64_t>() == 0 ||
        egress.empty())
      throw std::runtime_error(
          "functional nonlocal payload descriptor is incomplete");
    return;
  }
  const auto internal =
      payload_csr(payload, "/nonlocal/internal_transition");
  const auto emission = payload_csr(payload, "/nonlocal/emission");
  const auto losses = payload_csr(payload, "/nonlocal/internal_losses");
  const auto egress = payload_egress(payload);
  internal.validate();
  emission.validate();
  losses.validate();
  if (internal.rows != internal.cols || emission.rows != internal.rows ||
      losses.rows != internal.rows || emission.cols != egress.size())
    throw std::runtime_error(
        "nonlocal intrinsic block dimensions do not agree");
  for (std::uint64_t row = 0; row < internal.rows; ++row) {
    double total = 0.0;
    const auto accumulate_row = [row, &total](const CsrMatrix& matrix) {
      for (auto entry = matrix.indptr[row];
           entry < matrix.indptr[row + 1]; ++entry) {
        if (!std::isfinite(matrix.data[entry]) || matrix.data[entry] < 0.0)
          throw std::runtime_error(
              "nonlocal intrinsic block contains invalid weight");
        total += matrix.data[entry];
      }
    };
    accumulate_row(internal);
    accumulate_row(emission);
    accumulate_row(losses);
    if (std::abs(total - 1.0) > tolerance)
      throw std::runtime_error(
          "nonlocal intrinsic state row does not conserve probability");
  }
}

CustomRuntimeMap create_custom_runtimes(const Scene& scene,
                                        std::uint64_t local_state_count) {
  std::vector<std::uint32_t> surface_ids;
  for (const auto& [id, surface] : scene.surfaces)
    if (surface.kind == SurfaceKind::custom_local ||
        surface.kind == SurfaceKind::custom_nonlocal)
      surface_ids.push_back(id);
  std::sort(surface_ids.begin(), surface_ids.end());
  CustomRuntimeMap result;
  std::uint64_t offset = local_state_count;
  for (const auto id : surface_ids) {
    const auto& surface = scene.surfaces.at(id);
    CustomSurfaceRuntime runtime;
    runtime.plugin = std::make_shared<SurfacePlugin>(surface.plugin_path);
    runtime.config_json = surface.plugin_config_json;
    const auto expected =
        surface.kind == SurfaceKind::custom_local ? PluginLocality::local
                                                  : PluginLocality::nonlocal;
    if (runtime.plugin->locality() != expected)
      throw std::runtime_error("custom surface/plugin locality mismatch");
    if (expected == PluginLocality::nonlocal) {
      runtime.payload =
          runtime.plugin->build(runtime.config_json, scene.energy_eV);
      validate_intrinsic_nonlocal_payload(runtime.payload,
                                          scene.numerics.energy_tolerance);
      const auto config = nlohmann::json::parse(runtime.config_json);
      if (!config.contains("nonlocal_domain_id"))
        throw std::runtime_error(
            "nonlocal custom surface requires nonlocal_domain_id");
      runtime.nonlocal_domain_id =
          config.at("nonlocal_domain_id").get<std::int32_t>();
      if (!scene.media.count(runtime.nonlocal_domain_id))
        throw std::runtime_error(
            "nonlocal custom surface domain is not a declared medium");
      const auto metadata =
          nlohmann::json::parse(runtime.payload.metadata_json);
      runtime.functional =
          metadata.value("execution", std::string{}) == "function";
      if (runtime.functional) {
        FunctionOperator function(surface.plugin_path,
                                  runtime.config_json, scene.energy_eV);
        validate_function_operator(function,
                                   scene.numerics.energy_tolerance);
        runtime.function_descriptor = function.descriptor();
        runtime.state_count =
            runtime.function_descriptor.input_state_count;
        if (metadata.at("state_count").get<std::uint64_t>() !=
                runtime.state_count ||
            payload_egress(runtime.payload).size() !=
                runtime.function_descriptor.egress_count)
          throw std::runtime_error(
              "functional nonlocal payload and runtime dimensions disagree");
      } else {
        const auto transition =
            payload_csr(runtime.payload, "/nonlocal/internal_transition");
        if (transition.rows != transition.cols)
          throw std::runtime_error(
              "nonlocal plugin transition block must be square");
        runtime.state_count = transition.rows;
      }
      runtime.state_offset = offset;
      offset += runtime.state_count;
    }
    result.emplace(id, std::move(runtime));
  }
  return result;
}

}  // namespace

std::string OperatorBuilder::cache_key(const Scene& scene) {
  OperatorSet provenance;
  apply_provenance(provenance, scene);
  return provenance.cache_key_sha256;
}

OperatorSet OperatorBuilder::build(const Scene& scene) {
  SceneValidator::validate(scene).throw_if_invalid();
  Geometry geometry(scene);
  const auto local_basis = make_local_surface_basis(scene);
  const auto& state_emitters = local_basis.state_emitters;
  const auto& primitive_to_state = local_basis.primitive_to_state;
  std::vector<std::int32_t> channels;
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    if (!scene.mesh.triangle_transport.empty() &&
        !scene.mesh.triangle_transport.at(primitive))
      continue;
    const auto& surface = scene.surfaces.at(scene.mesh.surface_id[primitive]);
    if (surface.kind == SurfaceKind::sensitive ||
        (surface.kind == SurfaceKind::custom_local &&
         scene.mesh.channel_id[primitive] >= 0))
      channels.push_back(scene.mesh.channel_id[primitive]);
  }
  for (const auto& primitive : scene.mesh.analytic_primitives) {
    const auto& surface = scene.surfaces.at(primitive.surface_id);
    if (surface.kind == SurfaceKind::sensitive ||
        (surface.kind == SurfaceKind::custom_local &&
         primitive.channel_id >= 0))
      channels.push_back(primitive.channel_id);
  }
  const auto custom_runtimes =
      create_custom_runtimes(scene, state_emitters.size());
  std::uint64_t total_states = state_emitters.size();
  std::vector<std::string> state_labels = local_basis.state_labels;
  std::vector<std::string> loss_names{
      "bulk_absorption", "surface_absorption", "escape",
      "specular_or_model_truncation", "float32_intersection_miss"};
  std::unordered_map<std::uint32_t, std::uint64_t> plugin_loss_offset;
  for (const auto& [surface_id, runtime] : custom_runtimes) {
    if (runtime.plugin->locality() != PluginLocality::nonlocal) continue;
    const auto egress = payload_egress(runtime.payload);
    std::uint64_t loss_count = 0;
    if (runtime.functional) {
      if (runtime.function_descriptor.egress_count != egress.size())
        throw std::runtime_error(
            "functional nonlocal egress dimensions disagree");
      loss_count = runtime.function_descriptor.loss_count;
    } else {
      const auto emission =
          payload_csr(runtime.payload, "/nonlocal/emission");
      const auto losses =
          payload_csr(runtime.payload, "/nonlocal/internal_losses");
      if (emission.rows != runtime.state_count ||
          emission.cols != egress.size() ||
          losses.rows != runtime.state_count)
        throw std::runtime_error(
            "nonlocal plugin block row dimensions disagree");
      loss_count = losses.cols;
    }
    plugin_loss_offset[surface_id] = loss_names.size();
    for (const auto& name : payload_loss_names(runtime.payload, loss_count))
      loss_names.push_back(scene.surfaces.at(surface_id).name + "/" + name);
    total_states =
        std::max(total_states, runtime.state_offset + runtime.state_count);
  }
  state_labels.resize(total_states);
  for (const auto& [surface_id, runtime] : custom_runtimes)
    for (std::uint64_t state = 0; state < runtime.state_count; ++state)
      state_labels.at(runtime.state_offset + state) =
          scene.surfaces.at(surface_id).name + "/state:" +
          std::to_string(state);
  std::sort(channels.begin(), channels.end());
  channels.erase(std::unique(channels.begin(), channels.end()), channels.end());
  std::unordered_map<std::int32_t, std::uint32_t> channel_to_column;
  for (std::uint32_t i = 0; i < channels.size(); ++i)
    channel_to_column[channels[i]] = i;
  if (total_states == 0)
    throw std::runtime_error("scene has no transport states");
  if (channels.empty())
    throw std::runtime_error("scene has no sensitive channels");

  std::vector<std::map<std::uint32_t, double>> transition_rows(
      total_states);
  std::vector<std::map<std::uint32_t, double>> detection_rows(
      total_states);
  std::vector<std::map<std::uint32_t, double>> loss_rows(
      total_states);
  std::vector<FunctionBlock> function_blocks;
  const auto mu2_order = scene.numerics.lambertian_mu2_order;
  const auto phi_count = scene.numerics.lambertian_phi_count;
  const auto lambertian_mu2 = gauss_legendre_unit_interval(mu2_order);
  std::exception_ptr local_state_error;
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t state = 0;
       state < static_cast<std::int64_t>(state_emitters.size()); ++state) {
    try {
      RowAccumulation row;
      double state_area = 0.0;
      for (const auto& emitter : state_emitters[state])
        state_area += emitter.area_mm2;
      if (!(state_area > 0.0))
        throw std::runtime_error("surface basis has zero area");
      for (const auto& emitter : state_emitters[state]) {
        const double area_fraction = emitter.area_mm2 / state_area;
        const Vec3 geometric_normal = emitter.normal;
        const bool primary_is_minus =
            emitter.minus_domain_id == scene.primary_domain;
        const Vec3 inward =
            primary_is_minus ? scale(geometric_normal, -1.0)
                             : geometric_normal;
        const Vec3 tangent = emitter.tangent;
        const Vec3 bitangent = cross(inward, tangent);
        const Vec3 center = emitter.center;
        for (std::uint32_t radial = 0; radial < mu2_order; ++radial) {
          const double mu = std::sqrt(lambertian_mu2[radial].first);
          const double angular_weight =
              lambertian_mu2[radial].second / phi_count;
          const double transverse = std::sqrt(1.0 - mu * mu);
          for (std::uint32_t angular = 0; angular < phi_count; ++angular) {
            const double phi =
                2.0 * pi * (angular + 0.5) / phi_count;
            const Vec3 direction =
                normalized(add(
                    scale(inward, mu),
                    add(scale(tangent, transverse * std::cos(phi)),
                        scale(bitangent, transverse * std::sin(phi)))));
            trace_branch(
                scene, geometry, primitive_to_state, channel_to_column,
                custom_runtimes,
                {{add(center,
                      scale(inward, geometry.ray_origin_offset_mm())),
                  direction},
                 {area_fraction * angular_weight, 0, 0, 0}, tangent,
                 scene.primary_domain, 0, std::nullopt},
                row);
          }
        }
      }
      transition_rows[state] = std::move(row.transition);
      detection_rows[state] = std::move(row.detection);
      loss_rows[state] = std::move(row.losses);
    } catch (...) {
#pragma omp critical(oos_local_state_build_error)
      {
        if (!local_state_error)
          local_state_error = std::current_exception();
      }
    }
  }
  if (local_state_error) std::rethrow_exception(local_state_error);
  for (const auto& runtime_entry : custom_runtimes) {
    const auto surface_id = runtime_entry.first;
    const auto& runtime = runtime_entry.second;
    if (runtime.plugin->locality() != PluginLocality::nonlocal) continue;
    const auto egress = payload_egress(runtime.payload);
    std::vector<RowAccumulation> egress_rows(egress.size());
    std::exception_ptr egress_error;
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(egress.size()); ++index) {
      try {
        trace_branch(scene, geometry, primitive_to_state, channel_to_column,
                     custom_runtimes,
                     egress_ray(scene, surface_id, egress[index],
                                geometry.ray_origin_offset_mm()),
                     egress_rows[index]);
      } catch (...) {
#pragma omp critical(oos_egress_build_error)
        {
          if (!egress_error) egress_error = std::current_exception();
        }
      }
    }
    if (egress_error) std::rethrow_exception(egress_error);
    if (runtime.functional) {
      std::vector<std::map<std::uint32_t, double>> to_transition(
          egress.size());
      std::vector<std::map<std::uint32_t, double>> to_detection(
          egress.size());
      std::vector<std::map<std::uint32_t, double>> to_losses(
          egress.size());
      for (std::size_t index = 0; index < egress.size(); ++index) {
        to_transition[index] = std::move(egress_rows[index].transition);
        to_detection[index] = std::move(egress_rows[index].detection);
        to_losses[index] = std::move(egress_rows[index].losses);
      }
      FunctionBlock block;
      block.name = scene.surfaces.at(surface_id).name;
      block.library_path = scene.surfaces.at(surface_id).plugin_path;
      block.config_json = runtime.config_json;
      block.state_offset = runtime.state_offset;
      block.state_count = runtime.state_count;
      block.egress_count = egress.size();
      block.intrinsic_loss_count =
          runtime.function_descriptor.loss_count;
      block.contraction_bound =
          runtime.function_descriptor.contraction_bound;
      block.egress_to_transition =
          maps_to_csr(to_transition, total_states);
      block.egress_to_detection =
          maps_to_csr(to_detection, channels.size());
      block.egress_to_losses =
          maps_to_csr(to_losses, loss_names.size());
      for (std::uint64_t loss = 0;
           loss < block.intrinsic_loss_count; ++loss)
        block.intrinsic_loss_columns.push_back(
            static_cast<std::uint32_t>(
                plugin_loss_offset.at(surface_id) + loss));
      function_blocks.push_back(std::move(block));
      continue;
    }
    const auto internal =
        payload_csr(runtime.payload, "/nonlocal/internal_transition");
    const auto emission =
        payload_csr(runtime.payload, "/nonlocal/emission");
    const auto losses =
        payload_csr(runtime.payload, "/nonlocal/internal_losses");
    for (std::uint64_t row = 0; row < runtime.state_count; ++row) {
      const auto global_row =
          static_cast<std::uint32_t>(runtime.state_offset + row);
      for (std::uint64_t entry = internal.indptr[row];
           entry < internal.indptr[row + 1]; ++entry)
        transition_rows[global_row][static_cast<std::uint32_t>(
            runtime.state_offset + internal.indices[entry])] +=
            internal.data[entry];
      for (std::uint64_t entry = emission.indptr[row];
           entry < emission.indptr[row + 1]; ++entry) {
        const auto& traced = egress_rows.at(emission.indices[entry]);
        const double factor = emission.data[entry];
        for (const auto& [column, value] : traced.transition)
          transition_rows[global_row][column] += factor * value;
        for (const auto& [column, value] : traced.detection)
          detection_rows[global_row][column] += factor * value;
        for (const auto& [column, value] : traced.losses)
          loss_rows[global_row][column] += factor * value;
      }
      for (std::uint64_t entry = losses.indptr[row];
           entry < losses.indptr[row + 1]; ++entry)
        loss_rows[global_row][static_cast<std::uint32_t>(
            plugin_loss_offset.at(surface_id) + losses.indices[entry])] +=
            losses.data[entry];
    }
  }
  OperatorSet result;
  result.transition = maps_to_csr(transition_rows, total_states);
  result.detection = maps_to_csr(detection_rows, channels.size());
  result.losses = maps_to_csr(loss_rows, loss_names.size());
  result.function_blocks = std::move(function_blocks);
  result.state_labels = std::move(state_labels);
  result.channel_ids = std::move(channels);
  result.loss_names = std::move(loss_names);
  result.tolerance = scene.numerics.neumann_tolerance;
  result.maximum_iterations = scene.numerics.maximum_diffuse_bounces;
  result.ray_origin_offset_mm = geometry.ray_origin_offset_mm();
  apply_provenance(result, scene);
  result.validate();
  return result;
}

SourceBatch trace_source_quadratures(
    const Scene& scene, const OperatorSet& operators,
    const std::vector<SourceQuadrature>& quadratures) {
  operators.validate();
  Geometry geometry(scene);
  if (operators.ray_origin_offset_mm > 0.0 &&
      std::abs(geometry.ray_origin_offset_mm() -
               operators.ray_origin_offset_mm) >
          scene.numerics.energy_tolerance)
    throw std::runtime_error(
        "scene ray-origin offset does not match the operator cache");
  const auto local_basis = make_local_surface_basis(scene);
  const auto& primitive_to_state = local_basis.primitive_to_state;
  const std::uint32_t state = local_basis.state_emitters.size();
  const auto custom_runtimes = create_custom_runtimes(scene, state);
  std::uint64_t expected_states = state;
  for (const auto& [surface_id, runtime] : custom_runtimes) {
    (void)surface_id;
    expected_states =
        std::max(expected_states, runtime.state_offset + runtime.state_count);
  }
  if (expected_states != operators.transition.rows)
    throw std::runtime_error("scene states do not match operator cache");
  std::unordered_map<std::int32_t, std::uint32_t> channel_to_column;
  for (std::uint32_t i = 0; i < operators.channel_ids.size(); ++i)
    channel_to_column[operators.channel_ids[i]] = i;

  SourceBatch result;
  result.count = quadratures.size();
  result.initial_states.assign(result.count * operators.transition.rows, 0.0);
  result.direct_detection.assign(result.count * operators.detection.cols, 0.0);
  result.direct_losses.assign(result.count * operators.losses.cols, 0.0);
  result.source_integration_l1_error_estimate.assign(result.count, 0.0);
  const auto store_row = [&](std::size_t index,
                             const RowAccumulation& row) {
    for (const auto& [column, value] : row.transition)
      result.initial_states[index * operators.transition.rows + column] =
          value;
    for (const auto& [column, value] : row.detection)
      result.direct_detection[index * operators.detection.cols + column] =
          value;
    for (const auto& [loss, value] : row.losses) {
      if (loss >= operators.losses.cols)
        throw std::runtime_error("source trace produced an unknown loss");
      result.direct_losses[index * operators.losses.cols + loss] = value;
    }
  };

  // Ordinary ray quadratures parallelize naturally over batched sources.
  // Shape-factor sources instead parallelize over their hundreds of
  // thousands of boundary triangles, so they must not be enclosed by this
  // outer OpenMP region (nested OpenMP is disabled on production workers).
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(quadratures.size()); ++index) {
    if (quadratures[index].integration ==
        SourceIntegration::isotropic_surface_shape_factor)
      continue;
    RowAccumulation row;
    for (const auto& source_ray : quadratures[index].rays) {
      trace_branch(
          scene, geometry, primitive_to_state, channel_to_column,
          custom_runtimes,
          {source_ray.ray, source_ray.stokes, source_ray.reference_axis,
           source_ray.domain, 0, std::nullopt},
          row);
    }
    store_row(static_cast<std::size_t>(index), row);
  }

  std::vector<std::size_t> shape_factor_indices;
  std::map<double, ShapeFactorFeatureEdges> feature_edge_cache;
  for (std::size_t index = 0; index < quadratures.size(); ++index) {
    if (quadratures[index].integration !=
        SourceIntegration::isotropic_surface_shape_factor)
      continue;
    shape_factor_indices.push_back(index);
    if (quadratures[index].points.empty())
      throw std::runtime_error(
          "shape-factor source has no spatial quadrature points");
    const double dihedral =
        quadratures[index].shape_factor.feature_dihedral_degrees;
    auto found_features = feature_edge_cache.find(dihedral);
    if (found_features == feature_edge_cache.end())
      found_features =
          feature_edge_cache
              .emplace(dihedral,
                       build_shape_factor_feature_edges(scene, dihedral))
              .first;
  }

  const auto trace_shape_factor_quadrature =
      [&](std::size_t index) {
    RowAccumulation row;
    double estimated_l1_error = 0.0;
    const auto& feature_edges = feature_edge_cache.at(
        quadratures[index].shape_factor.feature_dihedral_degrees);
    for (const auto& source_point : quadratures[index].points) {
      auto integrated = trace_shape_factor_point(
          scene, geometry, primitive_to_state, channel_to_column,
          custom_runtimes, feature_edges, source_point,
          quadratures[index].shape_factor);
      add_row(row, integrated.row);
      estimated_l1_error += integrated.estimated_l1_error;
    }
    result.source_integration_l1_error_estimate[index] = estimated_l1_error;
    store_row(index, row);
  };

  if (shape_factor_indices.size() == 1) {
    // Preserve the lower-latency single-source path: the integration routine
    // parallelizes over boundary triangles.
    trace_shape_factor_quadrature(shape_factor_indices.front());
  } else if (!shape_factor_indices.empty()) {
    // Likelihood searches submit many independent candidate coordinates.
    // Parallelize those candidates at the outer level; nested OpenMP regions
    // in trace_shape_factor_point are serialized by the production runtime.
    // This keeps the same quadrature and hard depth cap while bounding the
    // wall-clock tail of a batched response request.
    std::exception_ptr shape_factor_error;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t offset = 0;
         offset < static_cast<std::int64_t>(shape_factor_indices.size());
         ++offset) {
      try {
        trace_shape_factor_quadrature(shape_factor_indices[offset]);
      } catch (...) {
#pragma omp critical(oos_shape_factor_batch_error)
        {
          if (!shape_factor_error)
            shape_factor_error = std::current_exception();
        }
      }
    }
    if (shape_factor_error) std::rethrow_exception(shape_factor_error);
  }
  return result;
}

}  // namespace oos
