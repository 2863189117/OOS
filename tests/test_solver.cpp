#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "oos/solver.hpp"

namespace {
oos::CsrMatrix csr(std::uint64_t rows, std::uint64_t cols,
                   std::vector<std::uint64_t> indptr,
                   std::vector<std::uint32_t> indices,
                   std::vector<double> data) {
  return {rows, cols, std::move(indptr), std::move(indices), std::move(data)};
}
}  // namespace

TEST_CASE("Neumann solve conserves detection and loss") {
  oos::OperatorSet operators;
  operators.transition = csr(1, 1, {0, 1}, {0}, {0.5});
  operators.detection = csr(1, 1, {0, 1}, {0}, {0.2});
  operators.losses = csr(1, 1, {0, 1}, {0}, {0.3});
  operators.channel_ids = {17};
  operators.loss_names = {"absorption"};
  operators.tolerance = 1e-13;
  oos::SourceBatch sources{1, {1.0}, {0.0}, {0.0}};
  const auto result = oos::Solver::solve_cpu(operators, sources);
  REQUIRE(result.efficiency[0] == Catch::Approx(0.4).margin(1e-12));
  REQUIRE(result.losses[0] == Catch::Approx(0.6).margin(1e-12));
  REQUIRE(result.unresolved[0] < 1e-13);
}

TEST_CASE("operator validation rejects energy creation") {
  oos::OperatorSet operators;
  operators.transition = csr(1, 1, {0, 1}, {0}, {0.9});
  operators.detection = csr(1, 1, {0, 1}, {0}, {0.2});
  operators.losses = csr(1, 1, {0, 0}, {}, {});
  operators.channel_ids = {0};
  REQUIRE_THROWS(operators.validate());
}

TEST_CASE("operator validation rejects a closed reflecting class") {
  oos::OperatorSet operators;
  operators.transition = {2, 2, {0, 1, 2}, {1, 0}, {1.0, 1.0}};
  operators.detection = {2, 1, {0, 0, 0}, {}, {}};
  operators.losses = {2, 1, {0, 0, 0}, {}, {}};
  operators.channel_ids = {0};
  operators.loss_names = {"absorption"};
  REQUIRE_THROWS_WITH(
      operators.validate(),
      Catch::Matchers::ContainsSubstring("non-terminating"));
}

TEST_CASE("float32 ray misses are reported as an efficiency-loss bound") {
  oos::OperatorSet operators;
  operators.transition = csr(1, 1, {0, 0}, {}, {});
  operators.detection = csr(1, 1, {0, 1}, {0}, {1.0});
  operators.losses = csr(1, 1, {0, 0}, {}, {});
  operators.channel_ids = {0};
  operators.loss_names = {"float32_intersection_miss"};
  operators.validate();

  oos::SourceBatch missed_source{1, {0.0}, {0.0}, {1.0e-3}};
  const auto direct = oos::Solver::solve_cpu(operators, missed_source);
  REQUIRE(direct.float32_efficiency_loss_upper_bound ==
          std::vector<double>{1.0e-3});

  operators.losses = csr(1, 1, {0, 1}, {0}, {1.0e-3});
  operators.detection = csr(1, 1, {0, 1}, {0}, {0.999});
  operators.validate();
  oos::SourceBatch propagated{1, {1.0}, {0.0}, {0.0}};
  const auto result = oos::Solver::solve_cpu(operators, propagated);
  REQUIRE(result.float32_efficiency_loss_upper_bound[0] ==
          Catch::Approx(1.0e-3));
}

TEST_CASE("Neumann solve composes a matrix-free functional block") {
  oos::OperatorSet operators;
  operators.transition = csr(2, 2, {0, 0, 0}, {}, {});
  operators.detection = csr(2, 1, {0, 0, 0}, {}, {});
  operators.losses = csr(2, 1, {0, 0, 0}, {}, {});
  operators.channel_ids = {17};
  operators.loss_names = {"intrinsic_absorption"};
  operators.energy_eV = 7.0;
  operators.tolerance = 1e-13;
  oos::FunctionBlock block;
  block.name = "test";
  block.library_path = OOS_TEST_FUNCTION_OPERATOR_PATH;
  block.config_json = "{}";
  block.state_offset = 0;
  block.state_count = 2;
  block.egress_count = 2;
  block.intrinsic_loss_count = 1;
  block.contraction_bound = 0.1;
  block.egress_to_transition = csr(2, 2, {0, 0, 0}, {}, {});
  block.egress_to_detection =
      csr(2, 1, {0, 1, 2}, {0, 0}, {1.0, 1.0});
  block.egress_to_losses = csr(2, 1, {0, 0, 0}, {}, {});
  block.intrinsic_loss_columns = {0};
  operators.function_blocks.push_back(std::move(block));
  oos::SourceBatch sources{1, {1.0, 0.0}, {0.0}, {0.0}};
  const auto result = oos::Solver::solve_cpu(operators, sources);
  REQUIRE(result.efficiency[0] ==
          Catch::Approx(2.0 / 3.0).margin(1e-12));
  REQUIRE(result.losses[0] == Catch::Approx(1.0 / 3.0).margin(1e-12));
  REQUIRE(result.unresolved[0] < 1e-13);
}
