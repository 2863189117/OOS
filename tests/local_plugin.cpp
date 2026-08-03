#include "oos/plugin.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <string>

namespace {
std::string message;

oos_string_view_v1 view(const std::string& value) {
  return {value.data(), value.size()};
}

nlohmann::json parse(oos_string_view_v1 value) {
  return nlohmann::json::parse(
      std::string(value.data ? value.data : "", value.size));
}

oos_plugin_validation_v1 validate(oos_string_view_v1 config,
                                  double energy_eV) {
  try {
    if (!(energy_eV > 0.0)) throw std::runtime_error("invalid energy");
    const auto value = parse(config);
    double sum = 0.0;
    for (const auto* name :
         {"specular_reflection", "specular_transmission",
          "lambertian_reflection", "detection", "absorption"}) {
      const double probability = value.value(name, 0.0);
      if (!std::isfinite(probability) || probability < 0.0)
        throw std::runtime_error("invalid local probability");
      sum += probability;
    }
    if (std::abs(sum - 1.0) > 1.0e-12)
      throw std::runtime_error("local probabilities do not close");
    message.clear();
    return {0, {nullptr, 0}};
  } catch (const std::exception& exception) {
    message = exception.what();
    return {1, view(message)};
  }
}

int32_t build(oos_string_view_v1, double,
              const oos_operator_sink_v1* sink) {
  const std::string metadata = R"({"locality":"local"})";
  return sink->write_metadata_json(sink->user_data, view(metadata));
}

int32_t interact(oos_string_view_v1 config, double,
                 const oos_surface_hit_v3*,
                 oos_local_interaction_v3* output) {
  try {
    const auto value = parse(config);
    *output = {
        value.value("specular_reflection", 0.0),
        value.value("specular_transmission", 0.0),
        value.value("lambertian_reflection", 0.0),
        value.value("detection", 0.0),
        value.value("absorption", 0.0),
    };
    return 0;
  } catch (...) {
    return 2;
  }
}

const oos_surface_plugin_v3 plugin{
    OOS_SURFACE_PLUGIN_ABI_V3,
    OOS_SURFACE_LOCAL_V3,
    {"oos_test_local", 14},
    {"1.0.0", 5},
    validate,
    build,
    interact,
    nullptr,
    0,
};
}  // namespace

extern "C" OOS_PLUGIN_EXPORT const oos_surface_plugin_v3*
oos_get_surface_plugin_v3(void) {
  return &plugin;
}
