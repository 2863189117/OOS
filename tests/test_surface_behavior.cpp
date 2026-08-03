#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "oos/surface_behavior.hpp"

namespace {
double total(const std::vector<oos::LocalSurfaceBranch>& branches) {
  double result = 0.0;
  for (const auto& branch : branches) result += branch.stokes.i;
  return result;
}
}  // namespace

TEST_CASE("Lambertian behavior is intrinsic and conservative") {
  oos::SurfaceModel surface;
  surface.kind = oos::SurfaceKind::lambertian;
  surface.reflectivity = 0.73;
  const auto branches = oos::evaluate_builtin_surface(
      surface, {{2.0, 0.4, -0.2, 0.1}, 1.0, 1.5, 0.25});
  REQUIRE(branches.size() == 2);
  REQUIRE(branches[0].kind ==
          oos::LocalBranchKind::lambertian_reflection);
  REQUIRE(branches[1].kind == oos::LocalBranchKind::absorption);
  REQUIRE(std::abs(total(branches) - 2.0) < 1.0e-15);
}

TEST_CASE("Fresnel behavior depends on local optical state only") {
  oos::SurfaceModel surface;
  surface.kind = oos::SurfaceKind::dielectric_fresnel;
  const oos::LocalSurfaceIncident incident{
      {1.0, 0.2, -0.1, 0.0}, 1.0, 1.6, 0.8};
  const auto first = oos::evaluate_builtin_surface(surface, incident);
  const auto relocated = oos::evaluate_builtin_surface(surface, incident);
  REQUIRE(first.size() == relocated.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    REQUIRE(first[i].kind == relocated[i].kind);
    REQUIRE(first[i].stokes.i == relocated[i].stokes.i);
    REQUIRE(first[i].stokes.q == relocated[i].stokes.q);
    REQUIRE(first[i].stokes.u == relocated[i].stokes.u);
    REQUIRE(first[i].stokes.v == relocated[i].stokes.v);
  }
  REQUIRE(std::abs(total(first) - 1.0) < 1.0e-12);
}

TEST_CASE("Opaque specular behavior is intrinsic and conservative") {
  oos::SurfaceModel surface;
  surface.kind = oos::SurfaceKind::specular_reflector;
  surface.reflectivity = 0.8;
  const auto branches = oos::evaluate_builtin_surface(
      surface, {{1.0, 0.2, -0.1, 0.0}, 1.0, 1.0, 0.6});
  REQUIRE(branches.size() == 2);
  REQUIRE(branches[0].kind == oos::LocalBranchKind::specular_reflection);
  REQUIRE(branches[1].kind == oos::LocalBranchKind::absorption);
  REQUIRE(branches[0].stokes.i == Catch::Approx(0.8));
  REQUIRE(std::abs(total(branches) - 1.0) < 1.0e-15);
}

TEST_CASE("Sensitive behavior leaves channel assignment to geometry") {
  oos::SurfaceModel surface;
  surface.kind = oos::SurfaceKind::sensitive;
  surface.detection_probability = 0.6;
  surface.remainder = oos::RemainderAction::reflect_lambertian;
  const auto branches = oos::evaluate_builtin_surface(
      surface, {{1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 1.0});
  REQUIRE(branches.size() == 2);
  REQUIRE(branches[0].kind == oos::LocalBranchKind::detection);
  REQUIRE(branches[1].kind ==
          oos::LocalBranchKind::lambertian_reflection);
  REQUIRE(std::abs(total(branches) - 1.0) < 1.0e-15);
}

TEST_CASE("Sensitive behavior can specularly reflect undetected light") {
  oos::SurfaceModel surface;
  surface.kind = oos::SurfaceKind::sensitive;
  surface.detection_probability = 0.8;
  surface.remainder = oos::RemainderAction::reflect_specular;
  const auto branches = oos::evaluate_builtin_surface(
      surface, {{1.0, 0.0, 0.0, 0.0}, 1.61, 3.4, 0.5});
  REQUIRE(branches.size() == 2);
  REQUIRE(branches[0].kind == oos::LocalBranchKind::detection);
  REQUIRE(branches[0].stokes.i == Catch::Approx(0.8));
  REQUIRE(branches[1].kind ==
          oos::LocalBranchKind::specular_reflection);
  REQUIRE(branches[1].stokes.i == Catch::Approx(0.2));
  REQUIRE(std::abs(total(branches) - 1.0) < 1.0e-15);
}
