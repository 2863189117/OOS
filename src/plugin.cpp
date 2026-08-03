#include "oos/plugin.hpp"

#include <dlfcn.h>

#include <stdexcept>

namespace oos {
namespace {
std::string to_string(oos_string_view_v1 value) {
  return std::string(value.data ? value.data : "", value.size);
}

oos_string_view_v1 view(const std::string& value) {
  return {value.data(), value.size()};
}

int32_t write_f64(void* user, oos_string_view_v1 name, const double* values,
                  const std::uint64_t* shape, std::size_t rank) {
  try {
    auto& output = *static_cast<PluginBuildResult*>(user);
    PluginDatasetF64 dataset;
    dataset.shape.assign(shape, shape + rank);
    std::size_t count = 1;
    for (const auto dimension : dataset.shape) count *= dimension;
    dataset.values.assign(values, values + count);
    output.f64.emplace(to_string(name), std::move(dataset));
    return 0;
  } catch (...) {
    return 1;
  }
}
int32_t write_u64(void* user, oos_string_view_v1 name,
                  const std::uint64_t* values, const std::uint64_t* shape,
                  std::size_t rank) {
  try {
    auto& output = *static_cast<PluginBuildResult*>(user);
    PluginDatasetU64 dataset;
    dataset.shape.assign(shape, shape + rank);
    std::size_t count = 1;
    for (const auto dimension : dataset.shape) count *= dimension;
    dataset.values.assign(values, values + count);
    output.u64.emplace(to_string(name), std::move(dataset));
    return 0;
  } catch (...) {
    return 1;
  }
}
int32_t write_metadata(void* user, oos_string_view_v1 value) {
  static_cast<PluginBuildResult*>(user)->metadata_json = to_string(value);
  return 0;
}
}  // namespace

SurfacePlugin::SurfacePlugin(const std::filesystem::path& path) {
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) throw std::runtime_error(dlerror());
  auto getter = reinterpret_cast<oos_get_surface_plugin_v3_fn>(
      dlsym(handle_, "oos_get_surface_plugin_v3"));
  if (!getter) {
    dlclose(handle_);
    handle_ = nullptr;
    throw std::runtime_error("plugin does not export oos_get_surface_plugin_v3");
  }
  api_ = getter();
  if (!api_ || api_->abi_version != OOS_SURFACE_PLUGIN_ABI_V3) {
    dlclose(handle_);
    handle_ = nullptr;
    throw std::runtime_error("surface plugin ABI mismatch");
  }
}

SurfacePlugin::~SurfacePlugin() {
  if (handle_) dlclose(handle_);
}

std::string SurfacePlugin::name() const { return to_string(api_->name); }
std::string SurfacePlugin::version() const { return to_string(api_->version); }
PluginLocality SurfacePlugin::locality() const {
  if (api_->locality == OOS_SURFACE_LOCAL_V3) return PluginLocality::local;
  if (api_->locality == OOS_SURFACE_NONLOCAL_V3)
    return PluginLocality::nonlocal;
  throw std::runtime_error("surface plugin declares an invalid locality");
}

void SurfacePlugin::validate(const std::string& config_json,
                             double energy_eV) const {
  oos_string_view_v1 config{config_json.data(), config_json.size()};
  const auto result = api_->validate(config, energy_eV);
  if (result.status != 0) throw std::runtime_error(to_string(result.message));
}

PluginBuildResult SurfacePlugin::build(const std::string& config_json,
                                       double energy_eV) const {
  validate(config_json, energy_eV);
  PluginBuildResult result;
  oos_operator_sink_v1 sink{&result, write_f64, write_u64, write_metadata};
  if (api_->build_operator(view(config_json), energy_eV, &sink) != 0)
    throw std::runtime_error("surface plugin operator build failed");
  return result;
}

oos_local_interaction_v3 SurfacePlugin::interact_local(
    const std::string& config_json, double energy_eV,
    const oos_surface_hit_v3& hit) const {
  if (locality() != PluginLocality::local || !api_->interact_local)
    throw std::runtime_error("surface plugin has no local interaction callback");
  oos_local_interaction_v3 result{};
  if (api_->interact_local(view(config_json), energy_eV, &hit, &result) != 0)
    throw std::runtime_error("local surface plugin interaction failed");
  return result;
}

NonlocalDeposition SurfacePlugin::deposit_nonlocal(
    const std::string& config_json, double energy_eV,
    const oos_surface_hit_v3& hit) const {
  if (locality() != PluginLocality::nonlocal || !api_->deposit_nonlocal)
    throw std::runtime_error("surface plugin has no nonlocal deposition callback");
  NonlocalDeposition result;
  result.state_indices.resize(api_->maximum_deposition_entries);
  result.weights.resize(api_->maximum_deposition_entries);
  std::size_t count = 0;
  if (api_->deposit_nonlocal(
          view(config_json), energy_eV, &hit, result.state_indices.data(),
          result.weights.data(), result.weights.size(), &count) != 0)
    throw std::runtime_error("nonlocal surface plugin deposition failed");
  if (count > result.weights.size())
    throw std::runtime_error("nonlocal plugin exceeded deposition capacity");
  result.state_indices.resize(count);
  result.weights.resize(count);
  return result;
}

}  // namespace oos
