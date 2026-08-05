#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "oos/analytic_geometry.hpp"
#include "oos/geometry.hpp"

TEST_CASE("Embree triangle scene returns surface identity") {
  oos::Scene scene;
  scene.mesh.vertices = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  scene.mesh.triangles = {{0, 1, 2}};
  scene.mesh.surface_id = {42};
  scene.mesh.minus_domain_id = {0};
  scene.mesh.plus_domain_id = {-1};
  scene.mesh.channel_id = {-1};
  oos::Geometry geometry(scene);
  const auto hit =
      geometry.intersect({{0, 0, 1}, {0, 0, -1}, 1e-9, 10.0});
  REQUIRE(hit.valid);
  REQUIRE(hit.surface_id == 42);
  REQUIRE(hit.distance == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("domain-aware intersection rejects nonadjacent geometry") {
  oos::Scene scene;
  scene.mesh.vertices = {
      {-1, -1, 1}, {1, -1, 1}, {0, 1, 1},
      {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  scene.mesh.triangles = {{0, 1, 2}, {3, 4, 5}};
  scene.mesh.surface_id = {10, 20};
  scene.mesh.minus_domain_id = {1, 0};
  scene.mesh.plus_domain_id = {-1, -1};
  scene.mesh.channel_id = {-1, -1};
  oos::Geometry geometry(scene);
  const oos::Ray ray{{0, 0, 2}, {0, 0, -1}, 1e-9, 10.0};

  REQUIRE(geometry.intersect(ray).surface_id == 10);
  const auto filtered = geometry.intersect(ray, 0);
  REQUIRE(filtered.valid);
  REQUIRE(filtered.surface_id == 20);
  REQUIRE(filtered.distance == Catch::Approx(2.0).margin(1e-6));
}

TEST_CASE("packet intersection preserves scalar triangle results") {
  oos::Scene scene;
  scene.mesh.vertices = {
      {-10, -10, 0}, {10, -10, 0}, {0, 10, 0}};
  scene.mesh.triangles = {{0, 1, 2}};
  scene.mesh.surface_id = {42};
  scene.mesh.minus_domain_id = {0};
  scene.mesh.plus_domain_id = {-1};
  scene.mesh.channel_id = {-1};
  oos::Geometry geometry(scene);
  std::vector<oos::Ray> rays(17,
      {{0, 0, 2}, {0, 0, -1}, 1.0e-9, 10.0});
  const auto packet = geometry.intersect_batch(rays, 0);
  REQUIRE(packet.size() == rays.size());
  for (std::size_t index = 0; index < rays.size(); ++index) {
    const auto scalar = geometry.intersect(rays[index], 0);
    REQUIRE(packet[index].valid == scalar.valid);
    CHECK(packet[index].geometry_key == scalar.geometry_key);
    CHECK(packet[index].distance ==
          Catch::Approx(scalar.distance).margin(1.0e-12));
  }
  const std::vector<std::int32_t> domains(rays.size(), 0);
  const auto mixed = geometry.intersect_batch(rays, domains);
  REQUIRE(mixed.size() == packet.size());
  for (std::size_t index = 0; index < mixed.size(); ++index)
    CHECK(mixed[index].geometry_key == packet[index].geometry_key);
  for (std::size_t index = 0; index < 8; ++index)
    rays[index].direction =
        index % 2 == 0 ? oos::Vec3{0, 0, -1}
                       : oos::Vec3{1, 0, -1};
  const auto divergent = geometry.intersect_batch(rays, 0);
  for (std::size_t index = 0; index < rays.size(); ++index)
    CHECK(divergent[index].valid ==
          geometry.intersect(rays[index], 0).valid);
}

namespace {
oos::Scene analytic_test_scene() {
  oos::Scene scene;
  // A validation/fallback triangle is retained in the scene, but transport
  // is owned by the exact analytic primitives.
  scene.mesh.vertices = {{-10, -10, -10}, {10, -10, -10}, {0, 10, -10}};
  scene.mesh.triangles = {{0, 1, 2}};
  scene.mesh.surface_id = {99};
  scene.mesh.surface_basis_id = {0};
  scene.mesh.minus_domain_id = {0};
  scene.mesh.plus_domain_id = {-1};
  scene.mesh.channel_id = {-1};
  scene.mesh.triangle_transport = {0};
  return scene;
}
}  // namespace

TEST_CASE("analytic disk and perforated disk use exact circular boundaries") {
  auto scene = analytic_test_scene();
  oos::AnalyticPrimitive disk;
  disk.kind = oos::GeometryPrimitiveKind::perforated_disk;
  disk.center_mm = {0, 0, 0};
  disk.parameters[0] = 5.0;
  disk.surface_id = 7;
  disk.minus_domain_id = 0;
  disk.plus_domain_id = -1;
  disk.surface_element = 12;
  disk.holes.push_back({{2.0, 0.0}, 0.5});
  scene.mesh.analytic_primitives.push_back(disk);
  oos::Geometry geometry(scene);

  const auto center =
      geometry.intersect({{0, 0, 2}, {0, 0, -1}}, 0);
  REQUIRE(center.valid);
  REQUIRE(center.kind == oos::GeometryPrimitiveKind::perforated_disk);
  REQUIRE(center.surface_id == 7);
  REQUIRE(center.surface_element == 12);
  REQUIRE(center.distance == Catch::Approx(2.0).margin(1e-12));
  REQUIRE_FALSE(
      geometry.intersect({{2.0, 0, 2}, {0, 0, -1}}, 0).valid);
  REQUIRE_FALSE(
      geometry.intersect({{5.01, 0, 2}, {0, 0, -1}}, 0).valid);

  std::vector<oos::Ray> rays(8,
      {{0, 0, 2}, {0, 0, -1}, 1.0e-9, 10.0});
  const auto packet = geometry.intersect_batch(rays, 0);
  for (const auto& packet_hit : packet) {
    REQUIRE(packet_hit.valid);
    CHECK(packet_hit.geometry_key == center.geometry_key);
    CHECK(packet_hit.distance ==
          Catch::Approx(center.distance).margin(1.0e-12));
  }
}

TEST_CASE("analytic finite cylinder returns its exact curved normal") {
  auto scene = analytic_test_scene();
  oos::AnalyticPrimitive cylinder;
  cylinder.kind = oos::GeometryPrimitiveKind::finite_cylinder;
  cylinder.parameters = {2.0, 3.0, 0.0, 0.0};
  cylinder.surface_id = 8;
  cylinder.minus_domain_id = 0;
  cylinder.plus_domain_id = -1;
  scene.mesh.analytic_primitives.push_back(cylinder);
  oos::Geometry geometry(scene);

  const auto hit =
      geometry.intersect({{4, 0, 1}, {-1, 0, 0}}, 0);
  REQUIRE(hit.valid);
  REQUIRE(hit.kind == oos::GeometryPrimitiveKind::finite_cylinder);
  REQUIRE(hit.distance == Catch::Approx(2.0).margin(1e-12));
  REQUIRE(hit.normal.x == Catch::Approx(1.0).margin(1e-12));
  REQUIRE(hit.normal.y == Catch::Approx(0.0).margin(1e-12));
  REQUIRE(hit.normal.z == Catch::Approx(0.0).margin(1e-12));
  REQUIRE_FALSE(
      geometry.intersect({{4, 0, 4}, {-1, 0, 0}}, 0).valid);
}

TEST_CASE("float broad-phase cylinder tangent advances after exact rejection") {
  auto scene = analytic_test_scene();
  oos::AnalyticPrimitive cylinder;
  cylinder.kind = oos::GeometryPrimitiveKind::finite_cylinder;
  cylinder.parameters = {1.0, 3.0, 0.0, 0.0};
  cylinder.surface_id = 8;
  cylinder.minus_domain_id = 0;
  cylinder.plus_domain_id = -1;
  scene.mesh.analytic_primitives.push_back(cylinder);
  oos::Geometry geometry(scene);

  // float32 rounds y to exactly one and reports a tangent candidate. The
  // double-precision ray lies outside the cylinder and must terminate without
  // recursively selecting the same broad-phase candidate.
  const auto hit =
      geometry.intersect({{-2.0, 1.0 + 1.0e-8, 0.0}, {1.0, 0.0, 0.0}}, 0);
  REQUIRE_FALSE(hit.valid);
}

TEST_CASE("analytic oriented box supports inside and outside rays") {
  auto scene = analytic_test_scene();
  oos::AnalyticPrimitive box;
  box.kind = oos::GeometryPrimitiveKind::box;
  box.center_mm = {1, 2, 3};
  box.parameters = {2.0, 3.0, 4.0, 0.0};
  box.surface_id = 9;
  box.minus_domain_id = 0;
  box.plus_domain_id = -1;
  scene.mesh.analytic_primitives.push_back(box);
  oos::Geometry geometry(scene);

  const auto outside =
      geometry.intersect({{5, 2, 3}, {-1, 0, 0}}, 0);
  REQUIRE(outside.valid);
  REQUIRE(outside.distance == Catch::Approx(2.0).margin(1e-12));
  REQUIRE(outside.normal.x == Catch::Approx(1.0).margin(1e-12));
  const auto inside =
      geometry.intersect({{1, 2, 3}, {0, 0, 1}}, 0);
  REQUIRE(inside.valid);
  REQUIRE(inside.distance == Catch::Approx(4.0).margin(1e-12));
  REQUIRE(inside.normal.z == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("analytic hits locate their independent radiance elements") {
  auto scene = analytic_test_scene();
  oos::AnalyticPrimitive disk;
  disk.kind = oos::GeometryPrimitiveKind::disk;
  disk.parameters[0] = 10.0;
  disk.surface_id = 0;
  disk.minus_domain_id = 0;
  disk.plus_domain_id = -1;
  scene.mesh.analytic_primitives.push_back(disk);
  oos::AnalyticSurfaceElement left;
  left.primitive_index = 0;
  left.bounds = {-10.0, 0.0, -10.0, 10.0, 0.0};
  left.center_mm = {-5.0, 0.0, 0.0};
  left.normal = {0.0, 0.0, 1.0};
  left.area_mm2 = 0.5 * 3.141592653589793 * 100.0;
  left.surface_basis_id = 4;
  left.surface_element = 8;
  auto right = left;
  right.bounds = {0.0, 10.0, -10.0, 10.0, 0.0};
  right.center_mm.x = 5.0;
  right.surface_basis_id = 5;
  right.surface_element = 9;
  scene.mesh.analytic_surface_elements = {left, right};

  oos::Geometry geometry(scene);
  const auto hit =
      geometry.intersect({{3.0, 0.0, 5.0}, {0.0, 0.0, -1.0}}, 0);
  REQUIRE(hit.valid);
  CHECK(hit.surface_basis_id == 5);
  CHECK(hit.surface_element == 9);
  CHECK(oos::is_analytic_surface_element_geometry_key(hit.geometry_key));
}
