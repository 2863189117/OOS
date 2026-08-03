#include "oos/physics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace oos {
namespace {
double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 scale(const Vec3& value, double factor) {
  return {factor * value.x, factor * value.y, factor * value.z};
}
}  // namespace

double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }

Vec3 normalized(const Vec3& value) {
  const double length = norm(value);
  if (!std::isfinite(length) || length == 0.0) {
    throw std::invalid_argument("vector must be finite and nonzero");
  }
  return scale(value, 1.0 / length);
}

FresnelResult fresnel_power(double n_incident, double n_transmitted,
                            double cos_incident) {
  if (!(n_incident > 0.0) || !(n_transmitted > 0.0)) {
    throw std::invalid_argument("refractive indices must be positive");
  }
  const double ci = std::clamp(cos_incident, 0.0, 1.0);
  const double sin2_t =
      std::pow(n_incident / n_transmitted, 2) * (1.0 - ci * ci);
  if (sin2_t >= 1.0) {
    return {0.0, true, {1.0, 1.0}, {0.0, 0.0}};
  }
  const double ct = std::sqrt(std::max(0.0, 1.0 - sin2_t));
  const double rs = (n_incident * ci - n_transmitted * ct) /
                    (n_incident * ci + n_transmitted * ct);
  const double rp = (n_transmitted * ci - n_incident * ct) /
                    (n_transmitted * ci + n_incident * ct);
  const std::array<double, 2> reflection{rs * rs, rp * rp};
  return {ct, false, reflection,
          {1.0 - reflection[0], 1.0 - reflection[1]}};
}

Stokes rotate_stokes(const Stokes& value, double angle_rad) {
  const double c = std::cos(2.0 * angle_rad);
  const double s = std::sin(2.0 * angle_rad);
  return {value.i, c * value.q + s * value.u,
          -s * value.q + c * value.u, value.v};
}

Stokes apply_linear_diattenuator(const Stokes& value, double s_power,
                                 double p_power) {
  if (s_power < 0.0 || p_power < 0.0) {
    throw std::invalid_argument("polarization powers cannot be negative");
  }
  const double sum = 0.5 * (s_power + p_power);
  const double difference = 0.5 * (s_power - p_power);
  const double cross = std::sqrt(s_power * p_power);
  return {sum * value.i + difference * value.q,
          difference * value.i + sum * value.q, cross * value.u,
          cross * value.v};
}

Vec3 reflect(const Vec3& direction, const Vec3& normal) {
  const Vec3 d = normalized(direction);
  const Vec3 n = normalized(normal);
  return normalized(add(d, scale(n, -2.0 * dot(d, n))));
}

Vec3 mirror_point_across_plane(const Vec3& point, const Vec3& plane_point,
                               const Vec3& unit_normal) {
  const Vec3 normal = normalized(unit_normal);
  const Vec3 offset{point.x - plane_point.x, point.y - plane_point.y,
                    point.z - plane_point.z};
  const double distance =
      offset.x * normal.x + offset.y * normal.y + offset.z * normal.z;
  return {point.x - 2.0 * distance * normal.x,
          point.y - 2.0 * distance * normal.y,
          point.z - 2.0 * distance * normal.z};
}

std::optional<Vec3> refract(const Vec3& direction,
                            const Vec3& normal_to_incident,
                            double n_incident, double n_transmitted) {
  const Vec3 d = normalized(direction);
  const Vec3 n = normalized(normal_to_incident);
  const double cos_i = -dot(d, n);
  const double eta = n_incident / n_transmitted;
  const double k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
  if (k < 0.0) return std::nullopt;
  return normalized(add(scale(d, eta),
                        scale(n, eta * cos_i - std::sqrt(k))));
}

}  // namespace oos
