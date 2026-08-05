#pragma once

#include <embree4/rtcore.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "oos/scene.hpp"

namespace oos {

class Geometry {
 public:
  explicit Geometry(const Scene& scene);
  ~Geometry();
  Geometry(const Geometry&) = delete;
  Geometry& operator=(const Geometry&) = delete;
  Geometry(Geometry&&) = delete;
  Geometry& operator=(Geometry&&) = delete;

  Hit intersect(const Ray& ray) const;
  Hit intersect(const Ray& ray, std::int32_t domain) const;
  // Intersect a coherent stream of rays that all originate in one domain.
  // Embree can vectorize/traverse this stream more efficiently than repeated
  // rtcIntersect1 calls. Results preserve input order.
  std::vector<Hit> intersect_batch(
      const std::vector<Ray>& rays, std::int32_t domain) const;
  // Mixed-domain convenience path. Rays are grouped by domain internally and
  // restored to caller order.
  std::vector<Hit> intersect_batch(
      const std::vector<Ray>& rays,
      const std::vector<std::int32_t>& domains) const;
  // Exact O(1) intersection against one declared analytic primitive. Used by
  // structured source integration after the geometry file has certified that
  // this primitive owns the first hit.
  std::optional<Hit> intersect_declared_analytic(
      std::uint32_t primitive_index, const Ray& ray,
      std::int32_t domain) const;
  bool has_nonadjacent_self_intersection() const;
  double ray_origin_offset_mm() const { return ray_origin_offset_mm_; }

 private:
  struct AnalyticElementGrid {
    AnalyticSurfaceCoordinates coordinates{};
    std::uint32_t primitive_index{};
    int box_face{-1};
    double first_min{};
    double second_min{};
    double first_step{1.0};
    double second_step{1.0};
    std::uint32_t first_count{1};
    std::uint32_t second_count{1};
    std::vector<std::vector<std::uint32_t>> cells;
  };

  std::optional<std::uint32_t> locate_analytic_surface_element(
      std::uint32_t primitive_index,
      const std::array<double, 3>& coordinates,
      const Vec3& normal) const;
  Hit decode_intersection(const Ray& ray, std::int32_t domain,
                          const RTCRayHit& query) const;

  const Scene* scene_{};
  RTCDevice device_{};
  RTCScene scene_handle_{};
  std::uint32_t triangle_geometry_id_{RTC_INVALID_GEOMETRY_ID};
  std::uint32_t analytic_geometry_id_{RTC_INVALID_GEOMETRY_ID};
  double ray_origin_offset_mm_{};
  std::vector<std::uint64_t> triangle_surface_element_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      analytic_elements_by_primitive_;
  std::unordered_map<std::uint64_t, AnalyticElementGrid>
      analytic_element_grids_;
};

}  // namespace oos
