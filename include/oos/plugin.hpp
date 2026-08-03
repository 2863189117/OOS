#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "oos/plugin.h"
#include "oos/types.hpp"

namespace oos {

struct PluginDatasetF64 {
  std::vector<double> values;
  std::vector<std::uint64_t> shape;
};
struct PluginDatasetU64 {
  std::vector<std::uint64_t> values;
  std::vector<std::uint64_t> shape;
};
struct PluginBuildResult {
  std::map<std::string, PluginDatasetF64> f64;
  std::map<std::string, PluginDatasetU64> u64;
  std::string metadata_json;
};

enum class PluginLocality {
  local,
  nonlocal,
};

struct NonlocalDeposition {
  std::vector<std::uint64_t> state_indices;
  std::vector<double> weights;
};

class SurfacePlugin {
 public:
  explicit SurfacePlugin(const std::filesystem::path& path);
  ~SurfacePlugin();
  SurfacePlugin(const SurfacePlugin&) = delete;
  SurfacePlugin& operator=(const SurfacePlugin&) = delete;

  std::string name() const;
  std::string version() const;
  PluginLocality locality() const;
  void validate(const std::string& config_json, double energy_eV) const;
  PluginBuildResult build(const std::string& config_json,
                          double energy_eV) const;
  oos_local_interaction_v3 interact_local(const std::string& config_json,
                                          double energy_eV,
                                          const oos_surface_hit_v3& hit) const;
  NonlocalDeposition deposit_nonlocal(const std::string& config_json,
                                      double energy_eV,
                                      const oos_surface_hit_v3& hit) const;
  const oos_surface_plugin_v3& api() const { return *api_; }

 private:
  void* handle_{};
  const oos_surface_plugin_v3* api_{};
};

}  // namespace oos
