#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

#include "oos/physics.hpp"

TEST_CASE("normal-incidence Fresnel power is conservative") {
  const auto result = oos::fresnel_power(1.0, 1.5, 1.0);
  REQUIRE_FALSE(result.total_internal_reflection);
  REQUIRE(result.reflectance[0] == Catch::Approx(0.04).margin(1e-15));
  REQUIRE(result.reflectance[1] == Catch::Approx(0.04).margin(1e-15));
  REQUIRE(result.reflectance[0] + result.transmittance[0] ==
          Catch::Approx(1.0).margin(1e-15));
}

TEST_CASE("total internal reflection returns unit reflected power") {
  const auto result = oos::fresnel_power(1.5, 1.0, 0.1);
  REQUIRE(result.total_internal_reflection);
  REQUIRE(result.reflectance[0] == 1.0);
  REQUIRE(result.transmittance[1] == 0.0);
}

TEST_CASE("Stokes rotation is reversible and conserves intensity") {
  const oos::Stokes input{1.0, 0.2, -0.3, 0.1};
  const auto rotated = oos::rotate_stokes(input, 0.37);
  const auto recovered = oos::rotate_stokes(rotated, -0.37);
  REQUIRE(rotated.i == input.i);
  REQUIRE(recovered.q == Catch::Approx(input.q).margin(1e-15));
  REQUIRE(recovered.u == Catch::Approx(input.u).margin(1e-15));
  REQUIRE(recovered.v == input.v);
}

TEST_CASE("reflection and refraction satisfy the planar geometry") {
  const oos::Vec3 incident{0.6, 0.0, -0.8};
  const auto reflected = oos::reflect(incident, {0.0, 0.0, 1.0});
  REQUIRE(reflected.z == Catch::Approx(0.8));
  const auto transmitted =
      oos::refract(incident, {0.0, 0.0, 1.0}, 1.0, 1.5);
  REQUIRE(transmitted.has_value());
  REQUIRE(oos::norm(*transmitted) == Catch::Approx(1.0));
  REQUIRE(transmitted->z < 0.0);
}

TEST_CASE("planar image source equals an explicitly reflected path") {
  const oos::Vec3 source{1.0, -2.0, 3.0};
  const oos::Vec3 target{-4.0, 3.0, 5.0};
  const oos::Vec3 normal{0.0, 0.0, 1.0};
  const auto image = oos::mirror_point_across_plane(
      source, oos::Vec3{0.0, 0.0, 0.0}, normal);
  const double fraction = -image.z / (target.z - image.z);
  const oos::Vec3 hit{image.x + fraction * (target.x - image.x),
                      image.y + fraction * (target.y - image.y), 0.0};
  const oos::Vec3 incoming = oos::normalized(
      {hit.x - source.x, hit.y - source.y, hit.z - source.z});
  const oos::Vec3 outgoing = oos::normalized(
      {target.x - hit.x, target.y - hit.y, target.z - hit.z});
  const auto reflected = oos::reflect(incoming, normal);
  REQUIRE(reflected.x == Catch::Approx(outgoing.x).margin(1e-12));
  REQUIRE(reflected.y == Catch::Approx(outgoing.y).margin(1e-12));
  REQUIRE(reflected.z == Catch::Approx(outgoing.z).margin(1e-12));
}
