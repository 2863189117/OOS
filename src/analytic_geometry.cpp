#include "oos/analytic_geometry.hpp"

#include "oos/physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace oos {
namespace {

Vec3 subtract(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 scale(const Vec3& value, double factor) {
  return {factor * value.x, factor * value.y, factor * value.z};
}

double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 local_vector(const AnalyticPrimitive& primitive, const Vec3& vector) {
  return {dot(vector, primitive.axis_x), dot(vector, primitive.axis_y),
          dot(vector, primitive.axis_z)};
}

Vec3 global_vector(const AnalyticPrimitive& primitive, const Vec3& vector) {
  return add(scale(primitive.axis_x, vector.x),
             add(scale(primitive.axis_y, vector.y),
                 scale(primitive.axis_z, vector.z)));
}

bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

bool adjacent(const AnalyticPrimitive& primitive, std::int32_t domain) {
  return domain == std::numeric_limits<std::int32_t>::min() ||
         domain == primitive.minus_domain_id ||
         domain == primitive.plus_domain_id;
}

bool accepted_distance(const Ray& ray, double distance) {
  return std::isfinite(distance) && distance >= ray.t_min &&
         distance <= ray.t_max;
}

std::optional<AnalyticIntersection> planar_intersection(
    const AnalyticPrimitive& primitive, const Ray& ray,
    std::uint32_t primitive_index) {
  const Vec3 local_origin =
      local_vector(primitive, subtract(ray.origin, primitive.center_mm));
  const Vec3 local_direction =
      local_vector(primitive, normalized(ray.direction));
  if (std::abs(local_direction.z) <=
      32.0 * std::numeric_limits<double>::epsilon())
    return std::nullopt;
  const double distance = -local_origin.z / local_direction.z;
  if (!accepted_distance(ray, distance)) return std::nullopt;
  const double u = local_origin.x + distance * local_direction.x;
  const double v = local_origin.y + distance * local_direction.y;
  const double radius2 = u * u + v * v;
  const double inner = primitive.kind == GeometryPrimitiveKind::annulus
                           ? primitive.parameters[0]
                           : 0.0;
  const double outer =
      primitive.kind == GeometryPrimitiveKind::annulus
          ? primitive.parameters[1]
          : primitive.parameters[0];
  if (radius2 < inner * inner || radius2 > outer * outer)
    return std::nullopt;
  if (primitive.kind == GeometryPrimitiveKind::perforated_disk)
    for (const auto& hole : primitive.holes) {
      const double du = u - hole.center_uv_mm.x;
      const double dv = v - hole.center_uv_mm.y;
      if (du * du + dv * dv < hole.radius_mm * hole.radius_mm)
        return std::nullopt;
    }
  const double normal_sign = primitive.normal_sign >= 0.0 ? 1.0 : -1.0;
  return AnalyticIntersection{
      primitive_index,
      distance,
      normalized(scale(primitive.axis_z, normal_sign)),
      {u, v, 0.0}};
}

std::optional<AnalyticIntersection> cylinder_intersection(
    const AnalyticPrimitive& primitive, const Ray& ray,
    std::uint32_t primitive_index) {
  const Vec3 local_origin =
      local_vector(primitive, subtract(ray.origin, primitive.center_mm));
  const Vec3 local_direction =
      local_vector(primitive, normalized(ray.direction));
  const double radius = primitive.parameters[0];
  const double half_length = primitive.parameters[1];
  const double a = local_direction.x * local_direction.x +
                   local_direction.y * local_direction.y;
  if (a <= 64.0 * std::numeric_limits<double>::epsilon())
    return std::nullopt;
  const double b = 2.0 * (local_origin.x * local_direction.x +
                          local_origin.y * local_direction.y);
  const double c = local_origin.x * local_origin.x +
                   local_origin.y * local_origin.y - radius * radius;
  const double discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) return std::nullopt;
  const double root = std::sqrt(std::max(0.0, discriminant));
  std::array<double, 2> distances{
      (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)};
  std::sort(distances.begin(), distances.end());
  for (const double distance : distances) {
    if (!accepted_distance(ray, distance)) continue;
    const double z = local_origin.z + distance * local_direction.z;
    if (std::abs(z) > half_length) continue;
    const double x = local_origin.x + distance * local_direction.x;
    const double y = local_origin.y + distance * local_direction.y;
    const double phi = std::atan2(y, x);
    const double normal_sign = primitive.normal_sign >= 0.0 ? 1.0 : -1.0;
    const Vec3 local_normal =
        scale(normalized({x, y, 0.0}), normal_sign);
    return AnalyticIntersection{
        primitive_index, distance,
        normalized(global_vector(primitive, local_normal)), {phi, z, 0.0}};
  }
  return std::nullopt;
}

std::optional<AnalyticIntersection> box_intersection(
    const AnalyticPrimitive& primitive, const Ray& ray,
    std::uint32_t primitive_index) {
  const Vec3 origin =
      local_vector(primitive, subtract(ray.origin, primitive.center_mm));
  const Vec3 direction =
      local_vector(primitive, normalized(ray.direction));
  const std::array<double, 3> o{origin.x, origin.y, origin.z};
  const std::array<double, 3> d{direction.x, direction.y, direction.z};
  const std::array<double, 3> half{
      primitive.parameters[0], primitive.parameters[1],
      primitive.parameters[2]};
  double near_distance = -std::numeric_limits<double>::infinity();
  double far_distance = std::numeric_limits<double>::infinity();
  int near_axis = -1;
  int far_axis = -1;
  double near_sign = 0.0;
  double far_sign = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(d[axis]) <=
        32.0 * std::numeric_limits<double>::epsilon()) {
      if (std::abs(o[axis]) > half[axis]) return std::nullopt;
      continue;
    }
    double first = (-half[axis] - o[axis]) / d[axis];
    double second = (half[axis] - o[axis]) / d[axis];
    double first_sign = -1.0;
    double second_sign = 1.0;
    if (first > second) {
      std::swap(first, second);
      std::swap(first_sign, second_sign);
    }
    if (first > near_distance) {
      near_distance = first;
      near_axis = axis;
      near_sign = first_sign;
    }
    if (second < far_distance) {
      far_distance = second;
      far_axis = axis;
      far_sign = second_sign;
    }
    if (near_distance > far_distance) return std::nullopt;
  }
  const bool use_near = accepted_distance(ray, near_distance);
  const double distance = use_near ? near_distance : far_distance;
  const int axis = use_near ? near_axis : far_axis;
  const double face_sign = use_near ? near_sign : far_sign;
  if (axis < 0 || !accepted_distance(ray, distance)) return std::nullopt;
  Vec3 local_normal{};
  if (axis == 0) local_normal.x = face_sign;
  if (axis == 1) local_normal.y = face_sign;
  if (axis == 2) local_normal.z = face_sign;
  const double normal_sign = primitive.normal_sign >= 0.0 ? 1.0 : -1.0;
  local_normal = scale(local_normal, normal_sign);
  const Vec3 point =
      {origin.x + distance * direction.x,
       origin.y + distance * direction.y,
       origin.z + distance * direction.z};
  return AnalyticIntersection{
      primitive_index, distance,
      normalized(global_vector(primitive, local_normal)),
      {point.x, point.y, point.z}};
}

}  // namespace

void validate_analytic_primitive(const AnalyticPrimitive& primitive,
                                 double tolerance) {
  if (!finite(primitive.center_mm) || !finite(primitive.axis_x) ||
      !finite(primitive.axis_y) || !finite(primitive.axis_z))
    throw std::runtime_error("analytic primitive frame is non-finite");
  const auto unit_error = [](const Vec3& value) {
    return std::abs(norm(value) - 1.0);
  };
  if (unit_error(primitive.axis_x) > tolerance ||
      unit_error(primitive.axis_y) > tolerance ||
      unit_error(primitive.axis_z) > tolerance ||
      std::abs(dot(primitive.axis_x, primitive.axis_y)) > tolerance ||
      std::abs(dot(primitive.axis_x, primitive.axis_z)) > tolerance ||
      std::abs(dot(primitive.axis_y, primitive.axis_z)) > tolerance)
    throw std::runtime_error(
        "analytic primitive frame must be orthonormal");
  const Vec3 handed{
      primitive.axis_x.y * primitive.axis_y.z -
          primitive.axis_x.z * primitive.axis_y.y,
      primitive.axis_x.z * primitive.axis_y.x -
          primitive.axis_x.x * primitive.axis_y.z,
      primitive.axis_x.x * primitive.axis_y.y -
          primitive.axis_x.y * primitive.axis_y.x};
  if (dot(handed, primitive.axis_z) < 1.0 - tolerance)
    throw std::runtime_error(
        "analytic primitive frame must be right-handed");
  if (!std::isfinite(primitive.normal_sign) ||
      std::abs(std::abs(primitive.normal_sign) - 1.0) > tolerance)
    throw std::runtime_error(
        "analytic primitive normal_sign must be +1 or -1");
  for (double parameter : primitive.parameters)
    if (!std::isfinite(parameter))
      throw std::runtime_error("analytic primitive parameter is non-finite");
  switch (primitive.kind) {
    case GeometryPrimitiveKind::disk:
    case GeometryPrimitiveKind::perforated_disk:
      if (!(primitive.parameters[0] > 0.0))
        throw std::runtime_error("analytic disk radius must be positive");
      break;
    case GeometryPrimitiveKind::annulus:
      if (!(primitive.parameters[0] >= 0.0) ||
          !(primitive.parameters[1] > primitive.parameters[0]))
        throw std::runtime_error(
            "analytic annulus radii are invalid");
      break;
    case GeometryPrimitiveKind::finite_cylinder:
      if (!(primitive.parameters[0] > 0.0) ||
          !(primitive.parameters[1] > 0.0))
        throw std::runtime_error(
            "analytic cylinder radius and half-length must be positive");
      break;
    case GeometryPrimitiveKind::box:
      if (!(primitive.parameters[0] > 0.0) ||
          !(primitive.parameters[1] > 0.0) ||
          !(primitive.parameters[2] > 0.0))
        throw std::runtime_error(
            "analytic box half-extents must be positive");
      break;
    case GeometryPrimitiveKind::triangle:
      throw std::runtime_error(
          "triangle is not a valid analytic primitive kind");
  }
  for (const auto& hole : primitive.holes)
    if (!std::isfinite(hole.center_uv_mm.x) ||
        !std::isfinite(hole.center_uv_mm.y) ||
        !(hole.radius_mm > 0.0) || !std::isfinite(hole.radius_mm))
      throw std::runtime_error("analytic disk hole is invalid");
}

std::optional<AnalyticIntersection> intersect_analytic_primitives(
    const std::vector<AnalyticPrimitive>& primitives, const Ray& ray,
    std::int32_t domain) {
  std::optional<AnalyticIntersection> closest;
  for (std::uint32_t index = 0; index < primitives.size(); ++index) {
    const auto candidate =
        intersect_analytic_primitive(primitives[index], index, ray, domain);
    if (candidate &&
        (!closest || candidate->distance < closest->distance))
      closest = candidate;
  }
  return closest;
}

std::optional<AnalyticIntersection> intersect_analytic_primitive(
    const AnalyticPrimitive& primitive, std::uint32_t primitive_index,
    const Ray& ray, std::int32_t domain) {
  if (!adjacent(primitive, domain)) return std::nullopt;
  switch (primitive.kind) {
    case GeometryPrimitiveKind::disk:
    case GeometryPrimitiveKind::annulus:
    case GeometryPrimitiveKind::perforated_disk:
      return planar_intersection(primitive, ray, primitive_index);
    case GeometryPrimitiveKind::finite_cylinder:
      return cylinder_intersection(primitive, ray, primitive_index);
    case GeometryPrimitiveKind::box:
      return box_intersection(primitive, ray, primitive_index);
    case GeometryPrimitiveKind::triangle:
      return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace oos
