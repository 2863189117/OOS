#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "oos/function_operator.hpp"

TEST_CASE("function operator validates and applies batches") {
  oos::FunctionOperator function(OOS_TEST_FUNCTION_OPERATOR_PATH, "{}", 7.0);
  oos::validate_function_operator(function);
  const auto result =
      function.apply_cpu(2, std::vector<double>{1.0, 2.0, 3.0, 4.0});
  const std::vector<double> expected_retained{0.1, 0.2, 0.3, 0.4};
  const std::vector<double> expected_egress{0.6, 1.2, 1.8, 2.4};
  const std::vector<double> expected_losses{0.9, 2.1};
  for (std::size_t index = 0; index < expected_retained.size(); ++index)
    REQUIRE(result.retained[index] ==
            Catch::Approx(expected_retained[index]).margin(1e-15));
  for (std::size_t index = 0; index < expected_egress.size(); ++index)
    REQUIRE(result.egress[index] ==
            Catch::Approx(expected_egress[index]).margin(1e-15));
  for (std::size_t index = 0; index < expected_losses.size(); ++index)
    REQUIRE(result.losses[index] ==
            Catch::Approx(expected_losses[index]).margin(1e-15));
  const auto adjoint = function.apply_adjoint_cpu(
      1, {2.0, 3.0}, {5.0, 7.0}, {11.0});
  REQUIRE(adjoint[0] ==
          Catch::Approx(0.1 * 2.0 + 0.6 * 5.0 + 0.3 * 11.0));
  REQUIRE(adjoint[1] ==
          Catch::Approx(0.1 * 3.0 + 0.6 * 7.0 + 0.3 * 11.0));
}
