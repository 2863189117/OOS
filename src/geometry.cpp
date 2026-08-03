#include "oos/geometry.hpp"

#include "oos/analytic_geometry.hpp"
#include "oos/physics.hpp"

#include <cmath>
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

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
bool segment_triangle(const Vec3& start, const Vec3& end, const Vec3& a,
                      const Vec3& b, const Vec3& c, double tolerance) {
  const Vec3 direction = subtract(end, start);
  const Vec3 edge1 = subtract(b, a);
  const Vec3 edge2 = subtract(c, a);
  const Vec3 p = cross(direction, edge2);
  const double determinant = dot(edge1, p);
  if (std::abs(determinant) <= tolerance) return false;
  const double inverse = 1.0 / determinant;
  const Vec3 t = subtract(start, a);
  const double u = dot(t, p) * inverse;
  if (u < -tolerance || u > 1.0 + tolerance) return false;
  const Vec3 q = cross(t, edge1);
  const double v = dot(direction, q) * inverse;
  if (v < -tolerance || u + v > 1.0 + tolerance) return false;
  const double parameter = dot(edge2, q) * inverse;
  return parameter >= -tolerance && parameter <= 1.0 + tolerance;
}
bool triangles_intersect(const std::array<Vec3, 3>& a,
                         const std::array<Vec3, 3>& b, double tolerance) {
  for (int edge = 0; edge < 3; ++edge) {
    if (segment_triangle(a[edge], a[(edge + 1) % 3], b[0], b[1], b[2],
                         tolerance) ||
        segment_triangle(b[edge], b[(edge + 1) % 3], a[0], a[1], a[2],
                         tolerance))
      return true;
  }
  return false;
}

double positive_phi(double value) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  value = std::fmod(value, two_pi);
  return value < 0.0 ? value + two_pi : value;
}

bool contains_half_open(double value, double lower, double upper,
                        double tolerance) {
  return value >= lower - tolerance && value < upper + tolerance;
}

int box_face_from_normal(const AnalyticPrimitive& primitive,
                         const Vec3& normal) {
  const Vec3 local{dot(normal, primitive.axis_x),
                   dot(normal, primitive.axis_y),
                   dot(normal, primitive.axis_z)};
  const std::array<double, 3> magnitude{
      std::abs(local.x), std::abs(local.y), std::abs(local.z)};
  const int axis = static_cast<int>(
      std::distance(magnitude.begin(),
                    std::max_element(magnitude.begin(), magnitude.end())));
  const double component =
      axis == 0 ? local.x : axis == 1 ? local.y : local.z;
  return 2 * axis + (component >= 0.0 ? 1 : 0);
}

std::uint64_t analytic_element_grid_key(
    std::uint32_t primitive_index,
    AnalyticSurfaceCoordinates coordinates, int box_face) {
  return (static_cast<std::uint64_t>(primitive_index) << 8u) |
         (static_cast<std::uint64_t>(coordinates) << 4u) |
         static_cast<std::uint64_t>(box_face + 1);
}

struct DomainQueryContext {
  RTCRayQueryContext context;
  const Scene* scene{};
  std::int32_t domain{std::numeric_limits<std::int32_t>::min()};
};

void adjacent_domain_filter(const RTCFilterFunctionNArguments* arguments) {
  if (arguments->N != 1 || !arguments->valid[0]) return;
  const auto* context =
      reinterpret_cast<const DomainQueryContext*>(arguments->context);
  const auto* hit = reinterpret_cast<const RTCHit*>(arguments->hit);
  const auto primitive = hit->primID;
  if (!context->scene->mesh.triangle_transport.empty() &&
      !context->scene->mesh.triangle_transport.at(primitive)) {
    arguments->valid[0] = 0;
    return;
  }
  if (context->domain == std::numeric_limits<std::int32_t>::min()) return;
  const auto minus = context->scene->mesh.minus_domain_id.at(primitive);
  const auto plus = context->scene->mesh.plus_domain_id.at(primitive);
  if (context->domain != minus && context->domain != plus)
    arguments->valid[0] = 0;
}

void analytic_bounds(const RTCBoundsFunctionArguments* arguments) {
  const auto* scene = static_cast<const Scene*>(arguments->geometryUserPtr);
  const auto& primitive =
      scene->mesh.analytic_primitives.at(arguments->primID);
  Vec3 extent{};
  switch (primitive.kind) {
    case GeometryPrimitiveKind::disk:
    case GeometryPrimitiveKind::annulus:
    case GeometryPrimitiveKind::perforated_disk: {
      const double radius =
          primitive.kind == GeometryPrimitiveKind::annulus
              ? primitive.parameters[1]
              : primitive.parameters[0];
      extent = {
          radius * std::hypot(primitive.axis_x.x, primitive.axis_y.x),
          radius * std::hypot(primitive.axis_x.y, primitive.axis_y.y),
          radius * std::hypot(primitive.axis_x.z, primitive.axis_y.z)};
      break;
    }
    case GeometryPrimitiveKind::finite_cylinder: {
      const double radius = primitive.parameters[0];
      const double half_length = primitive.parameters[1];
      extent = {
          radius * std::hypot(primitive.axis_x.x, primitive.axis_y.x) +
              half_length * std::abs(primitive.axis_z.x),
          radius * std::hypot(primitive.axis_x.y, primitive.axis_y.y) +
              half_length * std::abs(primitive.axis_z.y),
          radius * std::hypot(primitive.axis_x.z, primitive.axis_y.z) +
              half_length * std::abs(primitive.axis_z.z)};
      break;
    }
    case GeometryPrimitiveKind::box:
      extent = {
          primitive.parameters[0] * std::abs(primitive.axis_x.x) +
              primitive.parameters[1] * std::abs(primitive.axis_y.x) +
              primitive.parameters[2] * std::abs(primitive.axis_z.x),
          primitive.parameters[0] * std::abs(primitive.axis_x.y) +
              primitive.parameters[1] * std::abs(primitive.axis_y.y) +
              primitive.parameters[2] * std::abs(primitive.axis_z.y),
          primitive.parameters[0] * std::abs(primitive.axis_x.z) +
              primitive.parameters[1] * std::abs(primitive.axis_y.z) +
              primitive.parameters[2] * std::abs(primitive.axis_z.z)};
      break;
    case GeometryPrimitiveKind::triangle:
      return;
  }
  // Embree builds the broad-phase BVH in float32.  Give zero-thickness
  // analytic sheets a small conservative pad; the callback still performs
  // the exact double-precision acceptance test.
  constexpr double pad = 1.0e-3;
  arguments->bounds_o->lower_x =
      static_cast<float>(primitive.center_mm.x - extent.x - pad);
  arguments->bounds_o->lower_y =
      static_cast<float>(primitive.center_mm.y - extent.y - pad);
  arguments->bounds_o->lower_z =
      static_cast<float>(primitive.center_mm.z - extent.z - pad);
  arguments->bounds_o->upper_x =
      static_cast<float>(primitive.center_mm.x + extent.x + pad);
  arguments->bounds_o->upper_y =
      static_cast<float>(primitive.center_mm.y + extent.y + pad);
  arguments->bounds_o->upper_z =
      static_cast<float>(primitive.center_mm.z + extent.z + pad);
}

void analytic_intersect(
    const RTCIntersectFunctionNArguments* arguments) {
  if (arguments->N != 1 || !arguments->valid[0]) return;
  const auto* scene = static_cast<const Scene*>(arguments->geometryUserPtr);
  const auto* context =
      reinterpret_cast<const DomainQueryContext*>(arguments->context);
  auto* ray_hit = reinterpret_cast<RTCRayHit*>(arguments->rayhit);
  const Ray ray{{ray_hit->ray.org_x, ray_hit->ray.org_y,
                 ray_hit->ray.org_z},
                {ray_hit->ray.dir_x, ray_hit->ray.dir_y,
                 ray_hit->ray.dir_z},
                ray_hit->ray.tnear, ray_hit->ray.tfar};
  const auto candidate = intersect_analytic_primitive(
      scene->mesh.analytic_primitives.at(arguments->primID),
      arguments->primID, ray, context->domain);
  if (!candidate) return;
  ray_hit->ray.tfar = static_cast<float>(candidate->distance);
  ray_hit->hit.geomID = arguments->geomID;
  ray_hit->hit.primID = arguments->primID;
  ray_hit->hit.u = 0.0f;
  ray_hit->hit.v = 0.0f;
  ray_hit->hit.Ng_x = static_cast<float>(candidate->normal.x);
  ray_hit->hit.Ng_y = static_cast<float>(candidate->normal.y);
  ray_hit->hit.Ng_z = static_cast<float>(candidate->normal.z);
  for (unsigned level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level)
    ray_hit->hit.instID[level] = context->context.instID[level];
}
}  // namespace

Geometry::Geometry(const Scene& scene) : scene_(&scene) {
  double maximum_float_spacing = 0.0;
  const auto update_spacing = [&maximum_float_spacing](double value) {
    const float rounded = static_cast<float>(value);
    const float upward =
        std::nextafter(rounded, std::numeric_limits<float>::infinity());
    const float downward =
        std::nextafter(rounded, -std::numeric_limits<float>::infinity());
    maximum_float_spacing =
        std::max(maximum_float_spacing,
                 std::max(std::abs(static_cast<double>(upward) - rounded),
                          std::abs(static_cast<double>(rounded) - downward)));
  };
  for (const auto& vertex : scene.mesh.vertices) {
    update_spacing(vertex.x);
    update_spacing(vertex.y);
    update_spacing(vertex.z);
  }
  std::unordered_map<std::uint32_t, std::uint64_t> surface_counts;
  triangle_surface_element_.resize(scene.mesh.triangles.size());
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive)
    triangle_surface_element_[primitive] =
        surface_counts[scene.mesh.surface_id.at(primitive)]++;
  for (const auto& primitive : scene.mesh.analytic_primitives)
    validate_analytic_primitive(
        primitive, std::max(1.0e-12, scene.numerics.geometry_tolerance_mm));
  for (std::uint32_t element = 0;
       element < scene.mesh.analytic_surface_elements.size(); ++element)
    analytic_elements_by_primitive_[
        scene.mesh.analytic_surface_elements[element].primitive_index]
        .push_back(element);
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> grid_groups;
  for (std::uint32_t element = 0;
       element < scene.mesh.analytic_surface_elements.size(); ++element) {
    const auto& item = scene.mesh.analytic_surface_elements[element];
    const int face =
        item.coordinates == AnalyticSurfaceCoordinates::box_face_uv
            ? static_cast<int>(std::llround(item.bounds[4]))
            : -1;
    grid_groups[analytic_element_grid_key(
        item.primitive_index, item.coordinates, face)]
        .push_back(element);
  }
  for (const auto& [key, elements] : grid_groups) {
    AnalyticElementGrid grid;
    const auto& first = scene.mesh.analytic_surface_elements.at(
        elements.front());
    grid.coordinates = first.coordinates;
    grid.primitive_index = first.primitive_index;
    grid.box_face =
        first.coordinates == AnalyticSurfaceCoordinates::box_face_uv
            ? static_cast<int>(std::llround(first.bounds[4]))
            : -1;
    double first_max = -std::numeric_limits<double>::infinity();
    double second_max = -std::numeric_limits<double>::infinity();
    grid.first_min = std::numeric_limits<double>::infinity();
    grid.second_min = std::numeric_limits<double>::infinity();
    double minimum_first_width = std::numeric_limits<double>::infinity();
    double minimum_second_width = std::numeric_limits<double>::infinity();
    for (const auto element : elements) {
      const auto& item =
          scene.mesh.analytic_surface_elements.at(element);
      grid.first_min = std::min(grid.first_min, item.bounds[0]);
      first_max = std::max(first_max, item.bounds[1]);
      grid.second_min = std::min(grid.second_min, item.bounds[2]);
      second_max = std::max(second_max, item.bounds[3]);
      minimum_first_width =
          std::min(minimum_first_width,
                   item.bounds[1] - item.bounds[0]);
      minimum_second_width =
          std::min(minimum_second_width,
                   item.bounds[3] - item.bounds[2]);
    }
    const double first_extent = first_max - grid.first_min;
    const double second_extent = second_max - grid.second_min;
    grid.first_count = std::max<std::uint32_t>(
        1, static_cast<std::uint32_t>(
               std::ceil(first_extent / minimum_first_width)));
    grid.second_count = std::max<std::uint32_t>(
        1, static_cast<std::uint32_t>(
               std::ceil(second_extent / minimum_second_width)));
    constexpr std::uint64_t maximum_cells = 2'000'000;
    const std::uint64_t cells =
        static_cast<std::uint64_t>(grid.first_count) *
        grid.second_count;
    if (cells > maximum_cells) {
      const double scale_factor =
          std::sqrt(static_cast<double>(cells) / maximum_cells);
      grid.first_count = std::max<std::uint32_t>(
          1, static_cast<std::uint32_t>(
                 std::ceil(grid.first_count / scale_factor)));
      grid.second_count = std::max<std::uint32_t>(
          1, static_cast<std::uint32_t>(
                 std::ceil(grid.second_count / scale_factor)));
    }
    grid.first_step = first_extent / grid.first_count;
    grid.second_step = second_extent / grid.second_count;
    if (!(grid.first_step > 0.0) || !(grid.second_step > 0.0))
      throw std::runtime_error(
          "analytic surface-element grid has zero extent");
    grid.cells.resize(
        static_cast<std::size_t>(grid.first_count) *
        grid.second_count);
    const auto cell_index = [&grid](std::uint32_t first_index,
                                    std::uint32_t second_index) {
      return static_cast<std::size_t>(second_index) *
                 grid.first_count +
             first_index;
    };
    for (const auto element : elements) {
      const auto& item =
          scene.mesh.analytic_surface_elements.at(element);
      const auto index = [](double value, double minimum, double step,
                            std::uint32_t count) {
        return std::min<std::uint32_t>(
            count - 1,
            static_cast<std::uint32_t>(
                std::max(0.0, std::floor((value - minimum) / step))));
      };
      const auto first_begin =
          index(item.bounds[0], grid.first_min, grid.first_step,
                grid.first_count);
      const auto first_end =
          index(std::nextafter(item.bounds[1], item.bounds[0]),
                grid.first_min, grid.first_step, grid.first_count);
      const auto second_begin =
          index(item.bounds[2], grid.second_min, grid.second_step,
                grid.second_count);
      const auto second_end =
          index(std::nextafter(item.bounds[3], item.bounds[2]),
                grid.second_min, grid.second_step, grid.second_count);
      for (auto second_index = second_begin;
           second_index <= second_end; ++second_index)
        for (auto first_index = first_begin;
             first_index <= first_end; ++first_index)
          grid.cells[cell_index(first_index, second_index)]
              .push_back(element);
    }
    analytic_element_grids_.emplace(key, std::move(grid));
  }
  // Embree triangle vertices and rays use float32.  The previous
  // 10*geometry_tolerance offset could round back onto the just-hit surface
  // at metre-scale coordinates.  Sixty-four ULPs clears ordinary
  // grazing-edge ambiguity seen at triangulated aperture seams, while
  // remaining below the smallest declared scene layer thickness.
  ray_origin_offset_mm_ =
      std::max({10.0 * scene.numerics.geometry_tolerance_mm,
                scene.numerics.ray_origin_offset_mm,
                64.0 * maximum_float_spacing});
  device_ = rtcNewDevice(nullptr);
  if (!device_) throw std::runtime_error("failed to create Embree device");
  scene_handle_ = rtcNewScene(device_);
  rtcSetSceneFlags(scene_handle_, RTC_SCENE_FLAG_ROBUST);
  RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
  auto* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
      3 * sizeof(float), scene.mesh.vertices.size()));
  for (std::size_t i = 0; i < scene.mesh.vertices.size(); ++i) {
    vertices[3 * i] = static_cast<float>(scene.mesh.vertices[i].x);
    vertices[3 * i + 1] = static_cast<float>(scene.mesh.vertices[i].y);
    vertices[3 * i + 2] = static_cast<float>(scene.mesh.vertices[i].z);
  }
  auto* indices = static_cast<std::uint32_t*>(rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
      3 * sizeof(std::uint32_t), scene.mesh.triangles.size()));
  for (std::size_t i = 0; i < scene.mesh.triangles.size(); ++i) {
    indices[3 * i] = scene.mesh.triangles[i][0];
    indices[3 * i + 1] = scene.mesh.triangles[i][1];
    indices[3 * i + 2] = scene.mesh.triangles[i][2];
  }
  rtcCommitGeometry(geometry);
  triangle_geometry_id_ = rtcAttachGeometry(scene_handle_, geometry);
  rtcReleaseGeometry(geometry);
  if (!scene.mesh.analytic_primitives.empty()) {
    RTCGeometry analytic =
        rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(
        analytic, scene.mesh.analytic_primitives.size());
    rtcSetGeometryUserData(analytic, const_cast<Scene*>(&scene));
    rtcSetGeometryBoundsFunction(analytic, analytic_bounds, nullptr);
    rtcSetGeometryIntersectFunction(analytic, analytic_intersect);
    rtcCommitGeometry(analytic);
    analytic_geometry_id_ =
        rtcAttachGeometry(scene_handle_, analytic);
    rtcReleaseGeometry(analytic);
  }
  rtcCommitScene(scene_handle_);
}

Geometry::~Geometry() {
  if (scene_handle_) rtcReleaseScene(scene_handle_);
  if (device_) rtcReleaseDevice(device_);
}

Hit Geometry::intersect(const Ray& ray) const {
  return intersect(ray, std::numeric_limits<std::int32_t>::min());
}

Hit Geometry::intersect(const Ray& ray, std::int32_t domain) const {
  const Vec3 direction = normalized(ray.direction);
  RTCRayHit query{};
  query.ray.org_x = static_cast<float>(ray.origin.x);
  query.ray.org_y = static_cast<float>(ray.origin.y);
  query.ray.org_z = static_cast<float>(ray.origin.z);
  query.ray.dir_x = static_cast<float>(direction.x);
  query.ray.dir_y = static_cast<float>(direction.y);
  query.ray.dir_z = static_cast<float>(direction.z);
  query.ray.tnear = static_cast<float>(ray.t_min);
  query.ray.tfar = static_cast<float>(ray.t_max);
  query.ray.mask = 0xffffffffu;
  query.ray.flags = 0;
  query.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  query.hit.primID = RTC_INVALID_GEOMETRY_ID;
  RTCIntersectArguments arguments;
  rtcInitIntersectArguments(&arguments);
  DomainQueryContext context{};
  rtcInitRayQueryContext(&context.context);
  context.scene = scene_;
  context.domain = domain;
  arguments.context = &context.context;
  arguments.filter = adjacent_domain_filter;
  arguments.flags = static_cast<RTCRayQueryFlags>(
      RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER |
      RTC_RAY_QUERY_FLAG_INCOHERENT);
  rtcIntersect1(scene_handle_, &query, &arguments);
  if (query.hit.geomID == RTC_INVALID_GEOMETRY_ID) return {};
  Hit triangle_hit;
  if (query.hit.geomID == triangle_geometry_id_) {
    const std::uint32_t primitive = query.hit.primID;
    triangle_hit = {
        true,
        GeometryPrimitiveKind::triangle,
        primitive,
        primitive,
        scene_->mesh.surface_id.at(primitive),
        static_cast<double>(query.ray.tfar),
        normalized({query.hit.Ng_x, query.hit.Ng_y, query.hit.Ng_z}),
        scene_->mesh.minus_domain_id.at(primitive),
        scene_->mesh.plus_domain_id.at(primitive),
        scene_->mesh.channel_id.at(primitive),
        scene_->mesh.surface_basis_id.empty()
            ? primitive
            : scene_->mesh.surface_basis_id.at(primitive),
        triangle_surface_element_.at(primitive),
        {1.0 - query.hit.u - query.hit.v, query.hit.u, query.hit.v}};
    return triangle_hit;
  }
  if (query.hit.geomID != analytic_geometry_id_)
    throw std::runtime_error("Embree returned an unknown geometry ID");
  const auto analytic = intersect_analytic_primitive(
      scene_->mesh.analytic_primitives.at(query.hit.primID),
      query.hit.primID, ray, domain);
  if (!analytic) {
    // Embree traverses the conservative analytic bounds with float32 rays.
    // At a tangential circle/box edge it can therefore accept a user
    // primitive whose double-precision refinement correctly rejects the
    // ray. Continue just beyond that broad-phase candidate so the next exact
    // analytic or triangle boundary can win.
    Ray retry = ray;
    // The retry is converted back to float32 before Embree sees it. Advancing
    // by one double ULP can therefore round to the identical tnear and select
    // the same rejected candidate forever. Advance in Embree's precision and
    // require observable progress.
    retry.t_min = static_cast<double>(
        std::nextafter(query.ray.tfar,
                       std::numeric_limits<float>::infinity()));
    if (!(retry.t_min > ray.t_min) || !(retry.t_min <= retry.t_max))
      return {};
    return intersect(retry, domain);
  }
  const auto& primitive =
      scene_->mesh.analytic_primitives.at(analytic->primitive_index);
  const auto element = locate_analytic_surface_element(
      analytic->primitive_index, analytic->coordinates, analytic->normal);
  if (element) {
    const auto& basis =
        scene_->mesh.analytic_surface_elements.at(*element);
    return {
        true,
        primitive.kind,
        analytic_surface_element_geometry_key(*element),
        analytic->primitive_index,
        primitive.surface_id,
        analytic->distance,
        analytic->normal,
        primitive.minus_domain_id,
        primitive.plus_domain_id,
        primitive.channel_id,
        basis.surface_basis_id,
        basis.surface_element,
        analytic->coordinates};
  }
  return {
      true,
      primitive.kind,
      analytic_geometry_key(analytic->primitive_index),
      analytic->primitive_index,
      primitive.surface_id,
      analytic->distance,
      analytic->normal,
      primitive.minus_domain_id,
      primitive.plus_domain_id,
      primitive.channel_id,
      primitive.surface_basis_id,
      primitive.surface_element,
      analytic->coordinates};
}

std::optional<std::uint32_t> Geometry::locate_analytic_surface_element(
    std::uint32_t primitive_index,
    const std::array<double, 3>& coordinates,
    const Vec3& normal) const {
  const auto found = analytic_elements_by_primitive_.find(primitive_index);
  if (found == analytic_elements_by_primitive_.end()) return std::nullopt;
  const auto& primitive =
      scene_->mesh.analytic_primitives.at(primitive_index);
  const double tolerance =
      std::max(1.0e-12, scene_->numerics.geometry_tolerance_mm);
  double first = 0.0;
  double second = 0.0;
  int face = -1;
  const auto coordinate_kind =
      scene_->mesh.analytic_surface_elements.at(found->second.front())
          .coordinates;
  switch (coordinate_kind) {
    case AnalyticSurfaceCoordinates::plane_uv:
      first = coordinates[0];
      second = coordinates[1];
      break;
    case AnalyticSurfaceCoordinates::cylinder_phi_z:
      first = positive_phi(coordinates[0]);
      second = coordinates[1];
      break;
    case AnalyticSurfaceCoordinates::annulus_r2_phi:
      first =
          coordinates[0] * coordinates[0] +
          coordinates[1] * coordinates[1];
      second =
          positive_phi(std::atan2(coordinates[1], coordinates[0]));
      break;
    case AnalyticSurfaceCoordinates::box_face_uv:
      face = box_face_from_normal(primitive, normal);
      if (face / 2 == 0) {
        first = coordinates[1];
        second = coordinates[2];
      } else if (face / 2 == 1) {
        first = coordinates[0];
        second = coordinates[2];
      } else {
        first = coordinates[0];
        second = coordinates[1];
      }
      break;
  }
  const auto grid_found = analytic_element_grids_.find(
      analytic_element_grid_key(primitive_index, coordinate_kind, face));
  const std::vector<std::uint32_t>* candidates = &found->second;
  if (grid_found != analytic_element_grids_.end()) {
    const auto& grid = grid_found->second;
    if (first >= grid.first_min - tolerance &&
        second >= grid.second_min - tolerance) {
      const auto coordinate_index =
          [](double value, double minimum, double step,
             std::uint32_t count) {
            const double raw = std::floor((value - minimum) / step);
            if (raw < 0.0 || raw >= count)
              return std::optional<std::uint32_t>{};
            return std::optional<std::uint32_t>{
                static_cast<std::uint32_t>(raw)};
          };
      const auto first_index = coordinate_index(
          first, grid.first_min, grid.first_step, grid.first_count);
      const auto second_index = coordinate_index(
          second, grid.second_min, grid.second_step, grid.second_count);
      if (first_index && second_index)
        candidates = &grid.cells[
            static_cast<std::size_t>(*second_index) *
                grid.first_count +
            *first_index];
    }
  }
  for (const auto index : *candidates) {
    const auto& element =
        scene_->mesh.analytic_surface_elements.at(index);
    if (element.coordinates != coordinate_kind ||
        (coordinate_kind == AnalyticSurfaceCoordinates::box_face_uv &&
         face != static_cast<int>(std::llround(element.bounds[4]))))
      continue;
    if (contains_half_open(first, element.bounds[0], element.bounds[1],
                           tolerance) &&
        contains_half_open(second, element.bounds[2], element.bounds[3],
                           tolerance))
      return index;
  }
  if (primitive.kind == GeometryPrimitiveKind::perforated_disk &&
      coordinate_kind == AnalyticSurfaceCoordinates::plane_uv) {
    // The analytic top-plane quadtree estimates partially covered cells with
    // a finite midpoint mask.  Tiny exact-geometry slivers can consequently
    // lie outside every retained cell bound.  Its production lookup assigned
    // every such PTFE hit to the nearest retained top-patch centre; preserve
    // that ownership rule while keeping the intersection itself exact.
    std::optional<std::uint32_t> nearest;
    double best = std::numeric_limits<double>::infinity();
    for (const auto index : found->second) {
      const auto& element =
          scene_->mesh.analytic_surface_elements.at(index);
      if (element.coordinates !=
          AnalyticSurfaceCoordinates::plane_uv)
        continue;
      const Vec3 relative =
          subtract(element.center_mm, primitive.center_mm);
      const double du = dot(relative, primitive.axis_x) - first;
      const double dv = dot(relative, primitive.axis_y) - second;
      const double distance2 = du * du + dv * dv;
      if (distance2 < best) {
        best = distance2;
        nearest = index;
      }
    }
    if (nearest) return nearest;
  }
  return std::nullopt;
}

bool Geometry::has_nonadjacent_self_intersection() const {
  struct Bounds {
    std::uint32_t primitive{};
    Vec3 minimum;
    Vec3 maximum;
  };
  std::vector<Bounds> bounds;
  bounds.reserve(scene_->mesh.triangles.size());
  for (std::uint32_t primitive = 0;
       primitive < scene_->mesh.triangles.size(); ++primitive) {
    const auto& triangle = scene_->mesh.triangles[primitive];
    Bounds item{primitive,
                scene_->mesh.vertices[triangle[0]],
                scene_->mesh.vertices[triangle[0]]};
    for (int vertex = 1; vertex < 3; ++vertex) {
      const auto& point = scene_->mesh.vertices[triangle[vertex]];
      item.minimum.x = std::min(item.minimum.x, point.x);
      item.minimum.y = std::min(item.minimum.y, point.y);
      item.minimum.z = std::min(item.minimum.z, point.z);
      item.maximum.x = std::max(item.maximum.x, point.x);
      item.maximum.y = std::max(item.maximum.y, point.y);
      item.maximum.z = std::max(item.maximum.z, point.z);
    }
    bounds.push_back(item);
  }
  std::sort(bounds.begin(), bounds.end(),
            [](const Bounds& a, const Bounds& b) {
              return a.minimum.x < b.minimum.x;
            });
  const double tolerance = scene_->numerics.geometry_tolerance_mm;
  for (std::size_t i = 0; i < bounds.size(); ++i) {
    for (std::size_t j = i + 1; j < bounds.size(); ++j) {
      if (bounds[j].minimum.x > bounds[i].maximum.x + tolerance) break;
      if (bounds[j].minimum.y > bounds[i].maximum.y + tolerance ||
          bounds[j].maximum.y < bounds[i].minimum.y - tolerance ||
          bounds[j].minimum.z > bounds[i].maximum.z + tolerance ||
          bounds[j].maximum.z < bounds[i].minimum.z - tolerance)
        continue;
      const auto& first = scene_->mesh.triangles[bounds[i].primitive];
      const auto& second = scene_->mesh.triangles[bounds[j].primitive];
      bool adjacent = false;
      for (const auto a : first)
        for (const auto b : second) adjacent = adjacent || a == b;
      if (adjacent) continue;
      const std::array<Vec3, 3> first_points{
          scene_->mesh.vertices[first[0]], scene_->mesh.vertices[first[1]],
          scene_->mesh.vertices[first[2]]};
      const std::array<Vec3, 3> second_points{
          scene_->mesh.vertices[second[0]], scene_->mesh.vertices[second[1]],
          scene_->mesh.vertices[second[2]]};
      if (triangles_intersect(first_points, second_points, tolerance))
        return true;
    }
  }
  return false;
}

}  // namespace oos
