#include "oos/scene.hpp"

#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

#include "oos/hdf5_io.hpp"

namespace oos {
namespace {
SurfaceKind surface_kind(const std::string& value) {
  if (value == "dielectric_fresnel") return SurfaceKind::dielectric_fresnel;
  if (value == "specular_reflector") return SurfaceKind::specular_reflector;
  if (value == "lambertian") return SurfaceKind::lambertian;
  if (value == "sensitive") return SurfaceKind::sensitive;
  if (value == "custom_local") return SurfaceKind::custom_local;
  if (value == "custom_nonlocal") return SurfaceKind::custom_nonlocal;
  if (value == "custom_operator")
    throw std::runtime_error(
        "custom_operator was replaced by custom_local/custom_nonlocal");
  throw std::runtime_error("unknown surface kind " + value);
}
RemainderAction remainder_action(const std::string& value) {
  if (value == "absorb") return RemainderAction::absorb;
  if (value == "reflect_specular")
    return RemainderAction::reflect_specular;
  if (value == "reflect_lambertian")
    return RemainderAction::reflect_lambertian;
  if (value == "transmit") return RemainderAction::transmit;
  throw std::runtime_error("unknown remainder action " + value);
}
Vec3 vec3(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != 3) {
    throw std::runtime_error("expected a three-component vector");
  }
  return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}
}  // namespace

Scene Scene::load(const std::filesystem::path& yaml_path) {
  YAML::Node root = YAML::LoadFile(yaml_path.string());
  Scene result;
  result.source_path = std::filesystem::absolute(yaml_path);
  result.schema_version = root["schema_version"].as<std::uint32_t>();
  result.energy_eV = root["energy_eV"].as<double>();
  result.primary_domain = root["primary_domain"].as<std::int32_t>();
  result.primary_domain_seed_mm = vec3(root["primary_domain_seed_mm"]);
  const auto geometry_path =
      yaml_path.parent_path() / root["geometry_hdf5"].as<std::string>();
  result.geometry_path = std::filesystem::absolute(geometry_path);
  result.mesh = load_geometry_hdf5(geometry_path);

  for (const auto& node : root["media"]) {
    Medium medium;
    medium.id = node["id"].as<std::int32_t>();
    medium.name = node["name"].as<std::string>();
    medium.refractive_index = node["refractive_index"].as<double>();
    if (node["absorption_length_mm"]) {
      medium.absorption_length_mm =
          node["absorption_length_mm"].as<double>();
    }
    result.media.emplace(medium.id, std::move(medium));
  }
  for (const auto& node : root["surfaces"]) {
    SurfaceModel surface;
    surface.id = node["id"].as<std::uint32_t>();
    surface.name = node["name"].as<std::string>();
    surface.kind = surface_kind(node["model"].as<std::string>());
    const bool two_sided =
        node["two_sided"] && node["two_sided"].as<bool>();
    if (node["reflectivity"])
      surface.reflectivity = node["reflectivity"].as<double>();
    if (node["detection_probability"])
      surface.detection_probability =
          node["detection_probability"].as<double>();
    if (node["remainder"])
      surface.remainder =
          remainder_action(node["remainder"].as<std::string>());
    if (node["plugin"])
      surface.plugin_path =
          std::filesystem::absolute(yaml_path.parent_path() /
                                    node["plugin"].as<std::string>())
              .string();
    if (node["plugin_config_json"])
      surface.plugin_config_json =
          node["plugin_config_json"].as<std::string>();
    if (const auto frame = node["local_frame"]) {
      surface.frame_origin_mm = vec3(frame["origin_mm"]);
      surface.frame_x = vec3(frame["x_axis"]);
      surface.frame_y = vec3(frame["y_axis"]);
      surface.frame_z = vec3(frame["z_axis"]);
      surface.frame_declared = true;
    }
    if (!surface.plugin_config_json.empty()) {
      auto config = nlohmann::json::parse(surface.plugin_config_json);
      for (const auto* key :
           {"precomputed_operator_hdf5", "precomputed_block_hdf5",
            "factorized_block_hdf5"}) {
        if (!config.contains(key)) continue;
        auto operator_path =
            std::filesystem::path(
                config[key].get<std::string>());
        if (operator_path.is_relative())
          config[key] =
              std::filesystem::absolute(yaml_path.parent_path() / operator_path)
                  .string();
      }
      surface.plugin_config_json = config.dump();
    }
    surface.behavior_declared =
        two_sided &&
        ((surface.kind != SurfaceKind::lambertian &&
          surface.kind != SurfaceKind::specular_reflector) ||
         static_cast<bool>(node["reflectivity"])) &&
        (surface.kind != SurfaceKind::sensitive ||
         (static_cast<bool>(node["detection_probability"]) &&
          static_cast<bool>(node["remainder"]))) &&
        ((surface.kind != SurfaceKind::custom_local &&
          surface.kind != SurfaceKind::custom_nonlocal) ||
         (static_cast<bool>(node["plugin"]) &&
          static_cast<bool>(node["plugin_config_json"]) &&
          static_cast<bool>(node["local_frame"])));
    result.surfaces.emplace(surface.id, std::move(surface));
  }
  if (const auto numerics = root["numerics"]) {
    if (numerics["geometry_tolerance_mm"])
      result.numerics.geometry_tolerance_mm =
          numerics["geometry_tolerance_mm"].as<double>();
    if (numerics["ray_origin_offset_mm"])
      result.numerics.ray_origin_offset_mm =
          numerics["ray_origin_offset_mm"].as<double>();
    if (numerics["energy_tolerance"])
      result.numerics.energy_tolerance =
          numerics["energy_tolerance"].as<double>();
    if (numerics["neumann_tolerance"])
      result.numerics.neumann_tolerance =
          numerics["neumann_tolerance"].as<double>();
    if (numerics["lambertian_mu2_order"])
      result.numerics.lambertian_mu2_order =
          numerics["lambertian_mu2_order"].as<std::uint32_t>();
    if (numerics["lambertian_phi_count"])
      result.numerics.lambertian_phi_count =
          numerics["lambertian_phi_count"].as<std::uint32_t>();
    if (numerics["maximum_specular_hits"])
      result.numerics.maximum_specular_hits =
          numerics["maximum_specular_hits"].as<std::uint32_t>();
    if (numerics["maximum_diffuse_bounces"])
      result.numerics.maximum_diffuse_bounces =
          numerics["maximum_diffuse_bounces"].as<std::uint32_t>();
  }
  return result;
}

void Scene::apply_surface_basis(const std::filesystem::path& hdf5_path) {
  surface_basis_path = std::filesystem::absolute(hdf5_path);
  mesh.surface_basis_id =
      load_surface_basis_id_hdf5(surface_basis_path, mesh.triangles.size());
}

}  // namespace oos
