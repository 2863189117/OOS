#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace oos {

struct CsrMatrix {
  std::uint64_t rows{};
  std::uint64_t cols{};
  std::vector<std::uint64_t> indptr;
  std::vector<std::uint32_t> indices;
  std::vector<double> data;

  void validate() const;
  std::vector<double> right_multiply(const std::vector<double>& row) const;
};

struct FunctionBlock {
  std::string name;
  std::string library_path;
  std::string config_json;
  std::uint64_t state_offset{};
  std::uint64_t state_count{};
  std::uint64_t egress_count{};
  std::uint64_t intrinsic_loss_count{};
  double contraction_bound{};
  CsrMatrix egress_to_transition;
  CsrMatrix egress_to_detection;
  CsrMatrix egress_to_losses;
  std::vector<std::uint32_t> intrinsic_loss_columns;

  void validate(std::uint64_t global_states, std::uint64_t channels,
                std::uint64_t losses) const;
};

struct OperatorSet {
  CsrMatrix transition;
  CsrMatrix detection;
  CsrMatrix losses;
  std::vector<FunctionBlock> function_blocks;
  std::vector<std::string> state_labels;
  std::vector<std::int32_t> channel_ids;
  std::vector<std::string> loss_names;
  double tolerance{1e-10};
  std::uint32_t maximum_iterations{256};
  double ray_origin_offset_mm{};
  double energy_eV{};
  std::string cache_key_sha256;
  std::string scene_sha256;
  std::string geometry_sha256;
  std::string surface_basis_sha256;
  std::string dependency_lock_sha256;
  std::string code_commit;

  void validate() const;
};

struct SourceBatch {
  std::uint64_t count{};
  std::vector<double> initial_states;
  std::vector<double> direct_detection;
  std::vector<double> direct_losses;
  // A-posteriori L1 difference between the accepted shape-factor estimate
  // and its next coarser estimate. Zero for explicit ray quadratures.
  std::vector<double> source_integration_l1_error_estimate;
};

struct SolveResult {
  std::vector<double> efficiency;
  std::vector<double> losses;
  std::vector<double> unresolved;
  std::vector<double> input_weight;
  // Propagated from SourceBatch. This is an integration error estimate, not
  // a strict mathematical upper bound.
  std::vector<double> source_integration_l1_error_estimate;
  // Absolute upper bound on efficiency that could have been lost because an
  // otherwise enclosed ray missed the float32 Embree geometry.  This is the
  // cumulative float32_intersection_miss loss weight, not a fitted error.
  std::vector<double> float32_efficiency_loss_upper_bound;
  std::uint32_t iterations{};
  std::string backend;
  std::string hardware;
  double wall_seconds{};
  std::uint64_t peak_device_bytes{};
};

struct SolveControl {
  // Zero uses OperatorSet::maximum_iterations.
  std::uint32_t maximum_iterations{};
  // Production direct solves normally require convergence. Bounded effective
  // responses intentionally retain the unresolved remainder after a fixed
  // number of cycles.
  bool require_convergence{true};
};

class Solver {
 public:
  static SolveResult solve_cpu(const OperatorSet& operators,
                               const SourceBatch& sources,
                               SolveControl control = {});
};

void populate_float32_efficiency_loss_upper_bound(
    const OperatorSet& operators, std::uint64_t source_count,
    SolveResult& result);

}  // namespace oos
