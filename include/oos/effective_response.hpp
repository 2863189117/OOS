#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "oos/solver.hpp"

namespace oos {

// Materialized bounded Neumann response. Rows use the same global transport
// basis as OperatorSet, so source geometry remains continuous and is injected
// normally before this dense response is applied.
struct EffectiveResponse {
  std::uint64_t states{};
  std::uint64_t channels{};
  std::uint64_t losses{};
  std::uint32_t cycles{7};
  std::uint64_t build_batch_size{64};
  double operator_tolerance{1.0e-10};
  std::string construction_method{"adjoint_linear"};
  std::vector<double> state_to_detection;
  std::vector<double> state_to_losses;
  std::vector<double> state_unresolved;
  std::vector<std::int32_t> channel_ids;
  std::vector<std::string> loss_names;
  std::string operator_cache_key_sha256;
  std::string code_commit;

  void validate() const;
};

// Builds the bounded response by propagating terminal bases through the
// adjoint transport operator. There is intentionally no forward-basis
// precompute implementation.
EffectiveResponse build_effective_response_cpu(const OperatorSet& operators,
                                                std::uint32_t cycles = 7,
                                                std::uint64_t batch_size = 32);

SolveResult apply_effective_response_cpu(const EffectiveResponse& response,
                                         const SourceBatch& sources);

void save_effective_response_hdf5(const std::filesystem::path& path,
                                  const EffectiveResponse& response);
EffectiveResponse load_effective_response_hdf5(
    const std::filesystem::path& path);

}  // namespace oos
