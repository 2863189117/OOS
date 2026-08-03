#include <catch2/catch_test_macros.hpp>

#include "oos/validation.hpp"

#include <algorithm>
#include <limits>

namespace {
oos::Scene tetrahedron_scene() {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  scene.surfaces.emplace(
      0, oos::SurfaceModel{0, "wall", oos::SurfaceKind::lambertian, 0.8});
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id.assign(4, 0);
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, -1);
  scene.mesh.channel_id.assign(4, -1);
  return scene;
}
}  // namespace

TEST_CASE("closed oriented tetrahedron validates") {
  REQUIRE(oos::SceneValidator::validate(tetrahedron_scene()).ok());
}

TEST_CASE("missing face is rejected as an open domain") {
  auto scene = tetrahedron_scene();
  scene.mesh.triangles.pop_back();
  scene.mesh.surface_id.pop_back();
  scene.mesh.minus_domain_id.pop_back();
  scene.mesh.plus_domain_id.pop_back();
  scene.mesh.channel_id.pop_back();
  REQUIRE_FALSE(oos::SceneValidator::validate(scene).ok());
}

TEST_CASE("sensitive surface requires channels") {
  auto scene = tetrahedron_scene();
  scene.surfaces.at(0).kind = oos::SurfaceKind::sensitive;
  scene.surfaces.at(0).detection_probability = 1.0;
  REQUIRE_FALSE(oos::SceneValidator::validate(scene).ok());
}

TEST_CASE("reflective triangles must belong to the active surface basis") {
  auto scene = tetrahedron_scene();
  scene.mesh.surface_basis_id.assign(4, 0);
  scene.mesh.surface_basis_id[2] =
      std::numeric_limits<std::uint32_t>::max();

  const auto report = oos::SceneValidator::validate(scene);
  REQUIRE_FALSE(report.ok());
  REQUIRE(std::any_of(
      report.issues.begin(), report.issues.end(), [](const auto& issue) {
        return issue.code == "surface_basis";
      }));
}

TEST_CASE("absorbing triangles need not belong to the surface basis") {
  auto scene = tetrahedron_scene();
  scene.surfaces.at(0).reflectivity = 0.0;
  scene.mesh.surface_basis_id.assign(
      4, std::numeric_limits<std::uint32_t>::max());

  REQUIRE(oos::SceneValidator::validate(scene).ok());
}
