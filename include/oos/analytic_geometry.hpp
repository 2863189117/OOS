#pragma once

#include <optional>
#include <vector>

#include "oos/scene.hpp"

namespace oos {

struct AnalyticIntersection {
  std::uint32_t primitive_index{};
  double distance{};
  Vec3 normal{};
  std::array<double, 3> coordinates{1.0, 0.0, 0.0};
};

void validate_analytic_primitive(const AnalyticPrimitive& primitive,
                                 double tolerance);

std::optional<AnalyticIntersection> intersect_analytic_primitive(
    const AnalyticPrimitive& primitive, std::uint32_t primitive_index,
    const Ray& ray, std::int32_t domain);

std::optional<AnalyticIntersection> intersect_analytic_primitives(
    const std::vector<AnalyticPrimitive>& primitives, const Ray& ray,
    std::int32_t domain);

constexpr std::uint64_t analytic_geometry_key(std::uint32_t index) {
  return (std::uint64_t{1} << 63u) | index;
}

constexpr std::uint64_t analytic_surface_element_geometry_key(
    std::uint32_t index) {
  return (std::uint64_t{3} << 62u) | index;
}

constexpr bool is_analytic_surface_element_geometry_key(std::uint64_t key) {
  return (key >> 62u) == 3u;
}

}  // namespace oos
