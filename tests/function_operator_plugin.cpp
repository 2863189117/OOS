#include "oos/plugin.h"

#include <cstdint>
#include <numeric>

namespace {

oos_plugin_validation_v1 validate(oos_string_view_v1, double energy_eV) {
  static const char message[] = "energy must be positive";
  return energy_eV > 0.0
             ? oos_plugin_validation_v1{0, {nullptr, 0}}
             : oos_plugin_validation_v1{1, {message, sizeof(message) - 1}};
}

int32_t create(oos_string_view_v1, double, void** instance,
               oos_function_operator_descriptor_v1* descriptor) {
  static int token = 1;
  *instance = &token;
  *descriptor = {OOS_FUNCTION_OPERATOR_ABI_V1, 2, 2, 2, 1,
                 0.1, 1, 0};
  return 0;
}

void destroy(void*) {}

int32_t apply(void*, std::uint64_t batch, const double* input,
              double* retained, double* egress, double* losses,
              oos_function_operator_audit_v1* audit) {
  for (std::uint64_t row = 0; row < batch; ++row) {
    const double first = input[2 * row];
    const double second = input[2 * row + 1];
    retained[2 * row] = 0.1 * first;
    retained[2 * row + 1] = 0.1 * second;
    egress[2 * row] = 0.6 * first;
    egress[2 * row + 1] = 0.6 * second;
    losses[row] = 0.3 * (first + second);
    const double input_weight = first + second;
    audit[row] = {input_weight, 0.1 * input_weight,
                  0.6 * input_weight, 0.3 * input_weight, 0.0};
  }
  return 0;
}

const oos_function_operator_v1 function_operator{
    OOS_FUNCTION_OPERATOR_ABI_V1,
    {"oos_test_function", 17},
    {"1.0.0", 5},
    validate,
    create,
    destroy,
    apply,
    nullptr,
    nullptr,
};

}  // namespace

extern "C" OOS_PLUGIN_EXPORT const oos_function_operator_v1*
oos_get_function_operator_v1() {
  return &function_operator;
}
