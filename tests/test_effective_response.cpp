#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "oos/effective_response.hpp"
#include "oos/hdf5_io.hpp"

TEST_CASE("bounded effective response matches the direct Neumann cycles") {
  oos::OperatorSet operators;
  operators.transition = {1, 1, {0, 1}, {0}, {0.5}};
  operators.detection = {1, 1, {0, 1}, {0}, {0.2}};
  operators.losses = {1, 1, {0, 1}, {0}, {0.3}};
  operators.channel_ids = {17};
  operators.loss_names = {"absorption"};
  operators.cache_key_sha256 = "operator-key";
  operators.code_commit = "commit";
  operators.tolerance = 1.0e-13;

  const auto effective =
      oos::build_effective_response_cpu(operators, 2, 1);
  REQUIRE(effective.state_to_detection[0] ==
          Catch::Approx(0.3).margin(1.0e-14));
  REQUIRE(effective.state_to_losses[0] ==
          Catch::Approx(0.45).margin(1.0e-14));
  REQUIRE(effective.state_unresolved[0] ==
          Catch::Approx(0.25).margin(1.0e-14));

  oos::SourceBatch source{1, {0.8}, {0.1}, {0.05}};
  const auto materialized =
      oos::apply_effective_response_cpu(effective, source);
  const auto direct = oos::Solver::solve_cpu(
      operators, source, oos::SolveControl{2, false});
  REQUIRE(materialized.efficiency[0] ==
          Catch::Approx(direct.efficiency[0]).margin(1.0e-14));
  REQUIRE(materialized.losses[0] ==
          Catch::Approx(direct.losses[0]).margin(1.0e-14));
  REQUIRE(materialized.unresolved[0] ==
          Catch::Approx(direct.unresolved[0]).margin(1.0e-14));
}

TEST_CASE("effective response HDF5 round trip preserves its contract") {
  oos::EffectiveResponse value;
  value.states = 1;
  value.channels = 2;
  value.losses = 1;
  value.cycles = 7;
  value.state_to_detection = {0.2, 0.3};
  value.state_to_losses = {0.4};
  value.state_unresolved = {0.1};
  value.channel_ids = {3, 5};
  value.loss_names = {"absorption"};
  value.operator_cache_key_sha256 = "operator-key";
  value.code_commit = "commit";
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-effective-response-test.h5";
  oos::save_effective_response_hdf5(path, value);
  const auto restored = oos::load_effective_response_hdf5(path);
  REQUIRE(restored.cycles == 7);
  REQUIRE(restored.state_to_detection == value.state_to_detection);
  REQUIRE(restored.state_to_losses == value.state_to_losses);
  REQUIRE(restored.state_unresolved == value.state_unresolved);
  REQUIRE(restored.operator_cache_key_sha256 ==
          value.operator_cache_key_sha256);
  std::filesystem::remove(path);
}
