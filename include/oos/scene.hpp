#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "oos/types.hpp"

namespace oos {

struct AnalyticHole {
  Vec2 center_uv_mm;
  double radius_mm{};
};

// Declares that one exact analytic primitive replaces a complete collection
// of source-quadrature triangles.  `directional_disk` integrates the disk in
// solid-angle coordinates and therefore does not inherit the validation
// mesh's element count or tessellation quality.
enum class AnalyticSourceIntegral : std::uint8_t {
  none = 0,
  directional_disk = 1,
};

struct AnalyticPrimitive {
  GeometryPrimitiveKind kind{GeometryPrimitiveKind::disk};
  Vec3 center_mm;
  Vec3 axis_x{1.0, 0.0, 0.0};
  Vec3 axis_y{0.0, 1.0, 0.0};
  Vec3 axis_z{0.0, 0.0, 1.0};
  // disk: [outer_radius, 0, 0, 0]
  // annulus: [inner_radius, outer_radius, 0, 0]
  // finite_cylinder: [radius, half_length, 0, 0]
  // box: [half_x, half_y, half_z, 0]
  // perforated_disk: [outer_radius, 0, 0, 0] plus holes.
  std::array<double, 4> parameters{};
  double normal_sign{1.0};
  std::uint32_t surface_id{};
  std::uint32_t surface_basis_id{
      std::numeric_limits<std::uint32_t>::max()};
  std::int32_t minus_domain_id{-1};
  std::int32_t plus_domain_id{-1};
  std::int32_t channel_id{-1};
  std::uint64_t surface_element{};
  AnalyticSourceIntegral source_integral{AnalyticSourceIntegral::none};
  std::vector<AnalyticHole> holes;
};

enum class AnalyticSurfaceCoordinates : std::uint8_t {
  plane_uv = 0,
  cylinder_phi_z = 1,
  annulus_r2_phi = 2,
  box_face_uv = 3,
};

// Declares when source integration may replace a general first-hit query with
// a geometry-specific visibility rule. The declaration is explicit and
// fail-closed; only `direct` and `projected_aperture` bypass a BVH first hit.
enum class AnalyticSourceVisibility : std::uint8_t {
  ray_traced = 0,
  direct = 1,
  projected_aperture = 2,
};

struct AnalyticSurfaceElement {
  // Index into MeshData::analytic_primitives.  The primitive owns exact
  // intersection and domain adjacency; this element owns only the optical
  // quadrature/basis support on that surface.
  std::uint32_t primitive_index{};
  AnalyticSurfaceCoordinates coordinates{
      AnalyticSurfaceCoordinates::plane_uv};
  // Inclusive lower and exclusive upper coordinate bounds.  Periodic phi
  // intervals are normalized to [0,2*pi); box_face_uv stores face index in
  // bounds[4].
  std::array<double, 5> bounds{};
  Vec3 center_mm;
  Vec3 normal;
  double area_mm2{};
  std::uint32_t surface_basis_id{};
  std::uint64_t surface_element{};
  // Include this element in isotropic_surface_shape_factor source
  // integration.  Diffuse radiance elements normally set this flag; terminal
  // disks may additionally provide integration cells without owning states.
  bool source_quadrature{true};
  AnalyticSourceVisibility source_visibility{
      AnalyticSourceVisibility::ray_traced};
  // Optional finite-cell visibility aperture used only by source integration.
  // The referenced primitive must be a perforated disk and the hole index
  // selects one of its circular openings.  The optical element remains owned
  // by primitive_index; no external transport target is encoded here.
  std::uint32_t projected_aperture_primitive_index{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t projected_aperture_hole_index{
      std::numeric_limits<std::uint32_t>::max()};
};

struct MeshData {
  std::vector<Vec3> vertices;
  std::vector<std::array<std::uint32_t, 3>> triangles;
  std::vector<std::uint32_t> surface_id;
  // Geometry primitives and optical surface basis functions are independent.
  // The identifier is local to a surface.  Multiple triangles with the same
  // (surface_id, surface_basis_id, primary-domain side) share one radiance
  // state.
  std::vector<std::uint32_t> surface_basis_id;
  std::vector<std::int32_t> minus_domain_id;
  std::vector<std::int32_t> plus_domain_id;
  std::vector<std::int32_t> channel_id;
  // Zero marks a triangle as validation/fallback geometry only.  It remains
  // available to topology checks but is excluded from transport queries when
  // an exact analytic primitive represents the same physical surface.
  std::vector<std::uint8_t> triangle_transport;
  // Source integration and transport ownership are independent.  A triangle
  // can partition an exact analytic surface for adaptive solid-angle
  // integration without participating in runtime intersection.
  std::vector<std::uint8_t> triangle_source_quadrature;
  // Mapping from a source-quadrature triangle to the exact analytic
  // primitive that owns its first hit.  UINT32_MAX keeps the general BVH
  // visibility query.  This permits a validation mesh to provide adaptive
  // solid-angle cells while a disk/cylinder/box provides exact intersection.
  std::vector<std::uint32_t> triangle_source_analytic_primitive;
  std::vector<AnalyticPrimitive> analytic_primitives;
  std::vector<AnalyticSurfaceElement> analytic_surface_elements;
};

struct Numerics {
  double geometry_tolerance_mm{1e-8};
  // A value of zero selects an Embree-float-safe offset derived from the
  // geometry extent.  A positive value is treated as an additional lower
  // bound, independently of the topology tolerance above.
  double ray_origin_offset_mm{0.0};
  double energy_tolerance{1e-10};
  double neumann_tolerance{1e-10};
  std::uint32_t lambertian_mu2_order{12};
  std::uint32_t lambertian_phi_count{72};
  std::uint32_t maximum_specular_hits{64};
  std::uint32_t maximum_diffuse_bounces{256};
};

class Scene {
 public:
  static Scene load(const std::filesystem::path& yaml_path);
  void apply_surface_basis(const std::filesystem::path& hdf5_path);

  std::uint32_t schema_version{1};
  double energy_eV{};
  std::int32_t primary_domain{-1};
  Vec3 primary_domain_seed_mm{};
  MeshData mesh;
  Numerics numerics;
  std::unordered_map<std::int32_t, Medium> media;
  std::unordered_map<std::uint32_t, SurfaceModel> surfaces;
  std::filesystem::path source_path;
  std::filesystem::path geometry_path;
  std::filesystem::path surface_basis_path;
};

}  // namespace oos
