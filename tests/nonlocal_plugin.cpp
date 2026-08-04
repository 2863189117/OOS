#include "oos/plugin.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace {
std::string message;

oos_string_view_v1 view(const std::string& value) {
  return {value.data(), value.size()};
}

nlohmann::json parse(oos_string_view_v1 value) {
  return nlohmann::json::parse(
      std::string(value.data ? value.data : "", value.size));
}

oos_plugin_validation_v1 validate(oos_string_view_v1 config, double energy) {
  try {
    if (!(energy > 0.0)) throw std::runtime_error("invalid energy");
    (void)parse(config);
    message.clear();
    return {0, {nullptr, 0}};
  } catch (const std::exception& exception) {
    message = exception.what();
    return {1, view(message)};
  }
}

int32_t write_f64(const oos_operator_sink_v1* sink, const std::string& name,
                  const std::vector<double>& values,
                  const std::vector<std::uint64_t>& shape) {
  return sink->write_dataset_f64(sink->user_data, view(name), values.data(),
                                 shape.data(), shape.size());
}

int32_t write_u64(const oos_operator_sink_v1* sink, const std::string& name,
                  const std::vector<std::uint64_t>& values,
                  const std::vector<std::uint64_t>& shape) {
  return sink->write_dataset_u64(sink->user_data, view(name), values.data(),
                                 shape.data(), shape.size());
}

int32_t write_csr(const oos_operator_sink_v1* sink, const std::string& root,
                  double value) {
  if (write_u64(sink, root + "/shape", {1, 1}, {2}) ||
      write_u64(sink, root + "/indptr", {0, 1}, {2}) ||
      write_u64(sink, root + "/indices", {0}, {1}) ||
      write_f64(sink, root + "/data", {value}, {1}))
    return 1;
  return 0;
}

int32_t build(oos_string_view_v1 config, double,
              const oos_operator_sink_v1* sink) {
  try {
    const auto configuration = parse(config);
    // One intrinsic state emits 40% of its weight from the first element of
    // its own surface group.  No external primitive or channel is named.
    if ((!configuration.value("functional", false) &&
         (write_csr(sink, "/nonlocal/internal_transition", 0.1) ||
          write_csr(sink, "/nonlocal/emission", 0.4) ||
          write_csr(sink, "/nonlocal/internal_losses", 0.5))) ||
        write_u64(sink, "/nonlocal/egress/surface_element", {0}, {1}) ||
        write_f64(sink, "/nonlocal/egress/barycentric",
                  {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, {1, 3}) ||
        write_u64(sink, "/nonlocal/egress/side", {0}, {1}) ||
        write_f64(sink, "/nonlocal/egress/direction_local",
                  {0.0, 0.9578262852211513, 0.2873478855663454}, {1, 3}))
      return 1;
    if (configuration.value("emit_forbidden_channel", false) &&
        write_u64(sink, "/nonlocal/channel_id", {99}, {1}))
      return 1;
    const std::string metadata =
        configuration.value("functional", false)
            ? R"({"locality":"nonlocal","execution":"function","state_count":1,"loss_names":["internal_absorption"]})"
            : R"({"locality":"nonlocal","loss_names":["internal_absorption"]})";
    return sink->write_metadata_json(sink->user_data, view(metadata));
  } catch (...) {
    return 2;
  }
}

int32_t deposit(oos_string_view_v1, double, const oos_surface_hit_v3*,
                std::uint64_t* indices, double* weights,
                std::size_t capacity, std::size_t* count) {
  if (capacity < 1) return 1;
  indices[0] = 0;
  weights[0] = 1.0;
  *count = 1;
  return 0;
}

const oos_surface_plugin_v3 plugin{
    OOS_SURFACE_PLUGIN_ABI_V3,
    OOS_SURFACE_NONLOCAL_V3,
    {"oos_test_nonlocal", 17},
    {"1.0.0", 5},
    validate,
    build,
    nullptr,
    deposit,
    1,
};

int32_t create_function(oos_string_view_v1, double, void** instance,
                        oos_function_operator_descriptor_v2* descriptor) {
  static int token = 1;
  *instance = &token;
  *descriptor = {OOS_FUNCTION_OPERATOR_ABI_V2, 1, 1, 1, 1,
                 0.1, 1, 0};
  return 0;
}

void destroy_function(void*) {}

int32_t apply_function(void*, std::uint64_t batch, const double* input,
                       double* retained, double* egress, double* losses) {
  for (std::uint64_t row = 0; row < batch; ++row) {
    retained[row] = 0.1 * input[row];
    egress[row] = 0.4 * input[row];
    losses[row] = 0.5 * input[row];
  }
  return 0;
}

int32_t apply_function_adjoint(void*, std::uint64_t batch,
                               const double* retained,
                               const double* egress, const double* losses,
                               double* input) {
  for (std::uint64_t row = 0; row < batch; ++row)
    input[row] =
        0.1 * retained[row] + 0.4 * egress[row] + 0.5 * losses[row];
  return 0;
}

const oos_function_operator_v2 function_operator{
    OOS_FUNCTION_OPERATOR_ABI_V2,
    {"oos_test_nonlocal_function", 26},
    {"1.0.0", 5},
    validate,
    create_function,
    destroy_function,
    apply_function,
    apply_function_adjoint,
    nullptr,
    nullptr,
    nullptr,
};
}  // namespace

extern "C" OOS_PLUGIN_EXPORT const oos_surface_plugin_v3*
oos_get_surface_plugin_v3(void) {
  return &plugin;
}

extern "C" OOS_PLUGIN_EXPORT const oos_function_operator_v2*
oos_get_function_operator_v2(void) {
  return &function_operator;
}
