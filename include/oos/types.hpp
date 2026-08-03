#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace oos {

struct Vec3 {
  double x{};
  double y{};
  double z{};
};

struct Vec2 {
  double x{};
  double y{};
};

struct Ray {
  Vec3 origin;
  Vec3 direction;
  double t_min{1e-9};
  double t_max{std::numeric_limits<double>::infinity()};
};

struct Stokes {
  double i{1.0};
  double q{};
  double u{};
  double v{};
};

enum class SurfaceKind {
  dielectric_fresnel,
  specular_reflector,
  lambertian,
  sensitive,
  custom_local,
  custom_nonlocal,
};

enum class RemainderAction {
  absorb,
  reflect_specular,
  reflect_lambertian,
  transmit,
};

struct Medium {
  std::int32_t id{-1};
  std::string name;
  double refractive_index{};
  double absorption_length_mm{std::numeric_limits<double>::infinity()};
};

struct SurfaceModel {
  std::uint32_t id{};
  std::string name;
  SurfaceKind kind{};
  double reflectivity{};
  double detection_probability{};
  RemainderAction remainder{RemainderAction::absorb};
  std::string plugin_path;
  std::string plugin_config_json;
  Vec3 frame_origin_mm{};
  Vec3 frame_x{1.0, 0.0, 0.0};
  Vec3 frame_y{0.0, 1.0, 0.0};
  Vec3 frame_z{0.0, 0.0, 1.0};
  bool frame_declared{};
  bool behavior_declared{true};
};

enum class GeometryPrimitiveKind : std::uint8_t {
  triangle = 0,
  disk = 1,
  annulus = 2,
  finite_cylinder = 3,
  box = 4,
  perforated_disk = 5,
};

struct Hit {
  bool valid{};
  GeometryPrimitiveKind kind{GeometryPrimitiveKind::triangle};
  // Stable geometry key.  Triangle keys occupy [0, triangle_count); analytic
  // keys have their high bit set.  Transport-state maps use this key rather
  // than assuming that every hit is an Embree triangle.
  std::uint64_t geometry_key{};
  std::uint32_t primitive_id{};
  std::uint32_t surface_id{};
  double distance{};
  Vec3 normal{};
  std::int32_t minus_domain_id{-1};
  std::int32_t plus_domain_id{-1};
  std::int32_t channel_id{-1};
  std::uint32_t surface_basis_id{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint64_t surface_element{};
  // Triangles use ordinary barycentric coordinates.  Analytic primitives use
  // a documented surface-local parameterization but still provide a
  // conservative default (1,0,0) to plugins that only inspect local position.
  std::array<double, 3> barycentric{1.0, 0.0, 0.0};
};

}  // namespace oos
