#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hdf5.h>

#include <filesystem>

#include "oos/effective_response.hpp"
#include "oos/hdf5_io.hpp"

namespace {

oos::CsrMatrix csr(std::uint64_t rows, std::uint64_t cols,
                   std::vector<std::uint64_t> indptr,
                   std::vector<std::uint32_t> indices,
                   std::vector<double> data) {
  return {rows, cols, std::move(indptr), std::move(indices), std::move(data)};
}

oos::FunctionBlock function_block(std::string name, std::uint64_t offset,
                                  oos::CsrMatrix to_state,
                                  oos::CsrMatrix to_detection) {
  oos::FunctionBlock block;
  block.name = std::move(name);
  block.library_path = OOS_TEST_FUNCTION_OPERATOR_PATH;
  block.config_json = "{}";
  block.state_offset = offset;
  block.state_count = 2;
  block.egress_count = 2;
  block.intrinsic_loss_count = 1;
  block.contraction_bound = 0.1;
  block.egress_to_transition = std::move(to_state);
  block.egress_to_detection = std::move(to_detection);
  block.egress_to_losses = csr(2, 1, {0, 0, 0}, {}, {});
  block.intrinsic_loss_columns = {0};
  return block;
}

}  // namespace

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
  REQUIRE(effective.construction_method == "adjoint_linear");
}

TEST_CASE("automatic adjoint response converges the all-state remainder") {
  oos::OperatorSet operators;
  operators.transition = {1, 1, {0, 1}, {0}, {0.5}};
  operators.detection = {1, 1, {0, 1}, {0}, {0.2}};
  operators.losses = {1, 1, {0, 1}, {0}, {0.3}};
  operators.channel_ids = {17};
  operators.loss_names = {"absorption"};
  operators.cache_key_sha256 = "automatic-operator-key";
  operators.code_commit = "commit";
  operators.tolerance = 0.13;
  operators.maximum_iterations = 16;

  const auto effective =
      oos::build_effective_response_cpu(operators, 0, 1);
  REQUIRE(effective.cycles == 3);
  REQUIRE(effective.state_unresolved[0] ==
          Catch::Approx(0.125).margin(1.0e-14));

  oos::SourceBatch source{1, {0.8}, {0.1}, {0.05}};
  const auto materialized =
      oos::apply_effective_response_cpu(effective, source);
  const auto direct = oos::Solver::solve_cpu(operators, source);
  REQUIRE(materialized.iterations == direct.iterations);
  REQUIRE(materialized.efficiency[0] ==
          Catch::Approx(direct.efficiency[0]).margin(1.0e-14));
  REQUIRE(materialized.losses[0] ==
          Catch::Approx(direct.losses[0]).margin(1.0e-14));
  REQUIRE(materialized.unresolved[0] ==
          Catch::Approx(direct.unresolved[0]).margin(1.0e-14));

  operators.tolerance = 0.1;
  operators.maximum_iterations = 3;
  REQUIRE_THROWS_AS(oos::build_effective_response_cpu(operators, 0, 1),
                    std::runtime_error);
}

TEST_CASE("adjoint precompute composes multiple function blocks") {
  oos::OperatorSet operators;
  operators.transition = csr(4, 4, {0, 0, 0, 0, 0}, {}, {});
  operators.detection = csr(4, 1, {0, 0, 0, 0, 0}, {}, {});
  operators.losses = csr(4, 1, {0, 0, 0, 0, 0}, {}, {});
  operators.channel_ids = {17};
  operators.loss_names = {"intrinsic_absorption"};
  operators.energy_eV = 7.0;
  operators.tolerance = 1.0e-13;
  operators.cache_key_sha256 = "two-function-blocks";
  operators.function_blocks.push_back(function_block(
      "first", 0,
      csr(2, 4, {0, 1, 2}, {2, 3}, {1.0, 1.0}),
      csr(2, 1, {0, 0, 0}, {}, {})));
  operators.function_blocks.push_back(function_block(
      "second", 2, csr(2, 4, {0, 0, 0}, {}, {}),
      csr(2, 1, {0, 1, 2}, {0, 0}, {1.0, 1.0})));

  const auto effective =
      oos::build_effective_response_cpu(operators, 2, 2);
  oos::SourceBatch source{1, {1.0, 0.0, 0.0, 0.0}, {0.0}, {0.0}};
  const auto materialized =
      oos::apply_effective_response_cpu(effective, source);
  const auto direct = oos::Solver::solve_cpu(
      operators, source, oos::SolveControl{2, false});

  REQUIRE(materialized.efficiency[0] == Catch::Approx(0.36));
  REQUIRE(materialized.efficiency == direct.efficiency);
  REQUIRE(materialized.losses == direct.losses);
  REQUIRE(materialized.unresolved == direct.unresolved);
}

TEST_CASE("effective response HDF5 round trip preserves its contract") {
  oos::EffectiveResponse value;
  value.states = 1;
  value.channels = 2;
  value.losses = 1;
  value.cycles = 7;
  value.build_batch_size = 32;
  value.operator_tolerance = 1.0e-12;
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
  REQUIRE(restored.build_batch_size == 32);
  REQUIRE(restored.operator_tolerance == Catch::Approx(1.0e-12));
  REQUIRE(restored.construction_method == "adjoint_linear");
  REQUIRE(restored.state_to_detection == value.state_to_detection);
  REQUIRE(restored.state_to_losses == value.state_to_losses);
  REQUIRE(restored.state_unresolved == value.state_unresolved);
  REQUIRE(restored.operator_cache_key_sha256 ==
          value.operator_cache_key_sha256);
  REQUIRE(restored.fingerprint_sha256 ==
          oos::effective_response_fingerprint(value));

  const std::string obsolete_schema =
      "oos.effective-bounded-response.v1";
  const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const auto schema = H5Dopen2(file, "/metadata/schema", H5P_DEFAULT);
  REQUIRE(schema >= 0);
  REQUIRE(H5Dwrite(schema, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   obsolete_schema.data()) >= 0);
  H5Dclose(schema);
  H5Fclose(file);
  REQUIRE_THROWS_AS(oos::load_effective_response_hdf5(path),
                    std::runtime_error);
  std::filesystem::remove(path);
}
