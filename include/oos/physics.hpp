#pragma once

#include <array>
#include <optional>

#include "oos/types.hpp"

namespace oos {

struct FresnelResult {
  double cos_transmitted{};
  bool total_internal_reflection{};
  std::array<double, 2> reflectance{};
  std::array<double, 2> transmittance{};
};

FresnelResult fresnel_power(double n_incident, double n_transmitted,
                            double cos_incident);
Stokes rotate_stokes(const Stokes& value, double angle_rad);
Stokes apply_linear_diattenuator(const Stokes& value, double s_power,
                                 double p_power);
Vec3 reflect(const Vec3& direction, const Vec3& normal);
Vec3 mirror_point_across_plane(const Vec3& point, const Vec3& plane_point,
                               const Vec3& unit_normal);
std::optional<Vec3> refract(const Vec3& direction, const Vec3& normal_to_incident,
                            double n_incident, double n_transmitted);
double norm(const Vec3& value);
Vec3 normalized(const Vec3& value);

}  // namespace oos
