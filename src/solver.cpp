#include "oos/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>

#include "oos/function_operator.hpp"

namespace oos {

void CsrMatrix::validate() const {
  if (indptr.size() != rows + 1 || indices.size() != data.size() ||
      indptr.empty() || indptr.front() != 0 || indptr.back() != data.size()) {
    throw std::runtime_error("invalid CSR structure");
  }
  for (std::size_t i = 1; i < indptr.size(); ++i) {
    if (indptr[i] < indptr[i - 1])
      throw std::runtime_error("CSR indptr is not monotonic");
  }
  for (std::uint32_t index : indices) {
    if (index >= cols) throw std::runtime_error("CSR column out of range");
  }
  for (double value : data) {
    if (!std::isfinite(value) || value < 0.0)
      throw std::runtime_error("operator entries must be finite and nonnegative");
  }
}

std::vector<double> CsrMatrix::right_multiply(
    const std::vector<double>& row) const {
  if (row.size() != rows) throw std::runtime_error("matrix shape mismatch");
  std::vector<double> result(cols, 0.0);
  for (std::uint64_t i = 0; i < rows; ++i) {
    for (std::uint64_t entry = indptr[i]; entry < indptr[i + 1]; ++entry) {
      result[indices[entry]] += row[i] * data[entry];
    }
  }
  return result;
}

void FunctionBlock::validate(std::uint64_t global_states,
                             std::uint64_t channels,
                             std::uint64_t losses) const {
  if (name.empty() || library_path.empty() || state_count == 0 ||
      state_offset + state_count > global_states ||
      contraction_bound < 0.0 || contraction_bound >= 1.0)
    throw std::runtime_error("functional block descriptor is invalid");
  egress_to_transition.validate();
  egress_to_detection.validate();
  egress_to_losses.validate();
  if (egress_to_transition.rows != egress_count ||
      egress_to_transition.cols != global_states ||
      egress_to_detection.rows != egress_count ||
      egress_to_detection.cols != channels ||
      egress_to_losses.rows != egress_count ||
      egress_to_losses.cols != losses ||
      intrinsic_loss_columns.size() != intrinsic_loss_count)
    throw std::runtime_error("functional block coupling dimensions disagree");
  for (const auto column : intrinsic_loss_columns)
    if (column >= losses)
      throw std::runtime_error(
          "functional block intrinsic loss column is invalid");
}

void OperatorSet::validate() const {
  transition.validate();
  detection.validate();
  losses.validate();
  if (transition.rows != transition.cols ||
      detection.rows != transition.rows || losses.rows != transition.rows) {
    throw std::runtime_error("operator state dimensions disagree");
  }
  if (detection.cols != channel_ids.size() ||
      losses.cols != loss_names.size()) {
    throw std::runtime_error("operator terminal dimensions disagree");
  }
  if (!state_labels.empty() && state_labels.size() != transition.rows)
    throw std::runtime_error("operator state labels have wrong size");
  if (!std::isfinite(ray_origin_offset_mm) ||
      ray_origin_offset_mm < 0.0)
    throw std::runtime_error("operator ray-origin offset is invalid");
  std::vector<bool> function_state(transition.rows, false);
  for (const auto& block : function_blocks) {
    block.validate(transition.rows, detection.cols, losses.cols);
    for (std::uint64_t state = block.state_offset;
         state < block.state_offset + block.state_count; ++state) {
      if (function_state[state])
        throw std::runtime_error("functional block state ranges overlap");
      function_state[state] = true;
    }
  }
  for (std::uint64_t row = 0; row < transition.rows; ++row) {
    if (function_state[row]) {
      if (transition.indptr[row] != transition.indptr[row + 1] ||
          detection.indptr[row] != detection.indptr[row + 1] ||
          losses.indptr[row] != losses.indptr[row + 1])
        throw std::runtime_error(
            "functional states may not also have explicit operator rows");
      continue;
    }
    double total = 0.0;
    for (std::uint64_t i = transition.indptr[row];
         i < transition.indptr[row + 1]; ++i)
      total += transition.data[i];
    for (std::uint64_t i = detection.indptr[row];
         i < detection.indptr[row + 1]; ++i)
      total += detection.data[i];
    for (std::uint64_t i = losses.indptr[row]; i < losses.indptr[row + 1]; ++i)
      total += losses.data[i];
    if (total > 1.0 + tolerance)
      throw std::runtime_error("operator row creates energy");
  }
  std::vector<std::vector<std::uint32_t>> predecessors(transition.rows);
  std::vector<bool> reaches_terminal(transition.rows, false);
  std::queue<std::uint32_t> pending;
  for (std::uint32_t row = 0; row < transition.rows; ++row) {
    double terminal = 0.0;
    for (std::uint64_t i = detection.indptr[row];
         i < detection.indptr[row + 1]; ++i)
      terminal += detection.data[i];
    for (std::uint64_t i = losses.indptr[row]; i < losses.indptr[row + 1]; ++i)
      terminal += losses.data[i];
    if (terminal > tolerance) {
      reaches_terminal[row] = true;
      pending.push(row);
    }
    if (function_state[row] && !reaches_terminal[row]) {
      reaches_terminal[row] = true;
      pending.push(row);
    }
    for (std::uint64_t i = transition.indptr[row];
         i < transition.indptr[row + 1]; ++i)
      if (transition.data[i] > tolerance)
        predecessors[transition.indices[i]].push_back(row);
  }
  while (!pending.empty()) {
    const auto state = pending.front();
    pending.pop();
    for (const auto predecessor : predecessors[state]) {
      if (!reaches_terminal[predecessor]) {
        reaches_terminal[predecessor] = true;
        pending.push(predecessor);
      }
    }
  }
  if (std::any_of(reaches_terminal.begin(), reaches_terminal.end(),
                  [](bool value) { return !value; }))
    throw std::runtime_error(
        "operator contains a non-terminating perfectly reflecting class");
}

SolveResult Solver::solve_cpu(const OperatorSet& operators,
                              const SourceBatch& sources,
                              SolveControl control) {
  const auto started = std::chrono::steady_clock::now();
  operators.validate();
  const auto states = operators.transition.rows;
  const auto channels = operators.detection.cols;
  const auto loss_count = operators.losses.cols;
  if (sources.initial_states.size() != sources.count * states ||
      sources.direct_detection.size() != sources.count * channels ||
      sources.direct_losses.size() != sources.count * loss_count) {
    throw std::runtime_error("source batch shape mismatch");
  }
  SolveResult result;
  result.backend = "cpu";
#if defined(__VERSION__)
  result.hardware = std::string("native-cpu; compiler=") + __VERSION__;
#else
  result.hardware = "native-cpu";
#endif
  result.efficiency = sources.direct_detection;
  result.losses = sources.direct_losses;
  result.unresolved.assign(sources.count, 0.0);
  result.input_weight.assign(sources.count, 0.0);
  result.source_integration_l1_error_estimate =
      sources.source_integration_l1_error_estimate;
  if (result.source_integration_l1_error_estimate.empty())
    result.source_integration_l1_error_estimate.assign(sources.count, 0.0);
  if (result.source_integration_l1_error_estimate.size() != sources.count)
    throw std::runtime_error(
        "source integration error-estimate shape mismatch");
  std::vector<std::unique_ptr<FunctionOperator>> functions;
  functions.reserve(operators.function_blocks.size());
  for (const auto& block : operators.function_blocks) {
    auto function = std::make_unique<FunctionOperator>(
        block.library_path, block.config_json, operators.energy_eV);
    const auto& descriptor = function->descriptor();
    if (descriptor.input_state_count != block.state_count ||
        descriptor.retained_state_count != block.state_count ||
        descriptor.egress_count != block.egress_count ||
        descriptor.loss_count != block.intrinsic_loss_count ||
        std::abs(descriptor.contraction_bound - block.contraction_bound) >
            operators.tolerance)
      throw std::runtime_error(
          "functional block runtime descriptor does not match cache");
    validate_function_operator(*function, operators.tolerance);
    functions.push_back(std::move(function));
  }
  for (std::uint64_t source = 0; source < sources.count; ++source) {
    result.input_weight[source] =
        std::accumulate(sources.initial_states.begin() + source * states,
                        sources.initial_states.begin() + (source + 1) * states,
                        0.0) +
        std::accumulate(
            sources.direct_detection.begin() + source * channels,
            sources.direct_detection.begin() + (source + 1) * channels, 0.0) +
        std::accumulate(sources.direct_losses.begin() + source * loss_count,
                        sources.direct_losses.begin() +
                            (source + 1) * loss_count,
                        0.0);
    std::vector<double> current(
        sources.initial_states.begin() + source * states,
        sources.initial_states.begin() + (source + 1) * states);
    const auto maximum_iterations =
        control.maximum_iterations == 0 ? operators.maximum_iterations
                                        : control.maximum_iterations;
    if (maximum_iterations == 0)
      throw std::runtime_error("solve iteration limit must be positive");
    for (std::uint32_t iteration = 0; iteration < maximum_iterations;
         ++iteration) {
      const auto detected = operators.detection.right_multiply(current);
      const auto lost = operators.losses.right_multiply(current);
      for (std::uint64_t channel = 0; channel < channels; ++channel)
        result.efficiency[source * channels + channel] += detected[channel];
      for (std::uint64_t loss = 0; loss < loss_count; ++loss)
        result.losses[source * loss_count + loss] += lost[loss];
      auto next = operators.transition.right_multiply(current);
      for (std::size_t block_index = 0;
           block_index < operators.function_blocks.size(); ++block_index) {
        const auto& block = operators.function_blocks[block_index];
        std::vector<double> input(
            current.begin() + block.state_offset,
            current.begin() + block.state_offset + block.state_count);
        const auto applied = functions[block_index]->apply_cpu(1, input);
        for (std::uint64_t state = 0; state < block.state_count; ++state)
          next[block.state_offset + state] += applied.retained[state];
        const auto coupled_state =
            block.egress_to_transition.right_multiply(applied.egress);
        const auto coupled_detection =
            block.egress_to_detection.right_multiply(applied.egress);
        const auto coupled_losses =
            block.egress_to_losses.right_multiply(applied.egress);
        for (std::uint64_t state = 0; state < states; ++state)
          next[state] += coupled_state[state];
        for (std::uint64_t channel = 0; channel < channels; ++channel)
          result.efficiency[source * channels + channel] +=
              coupled_detection[channel];
        for (std::uint64_t loss = 0; loss < loss_count; ++loss)
          result.losses[source * loss_count + loss] += coupled_losses[loss];
        for (std::uint64_t loss = 0;
             loss < block.intrinsic_loss_count; ++loss)
          result.losses[source * loss_count +
                        block.intrinsic_loss_columns[loss]] +=
              applied.losses[loss];
      }
      current = std::move(next);
      const double unresolved =
          std::accumulate(current.begin(), current.end(), 0.0);
      result.unresolved[source] = unresolved;
      result.iterations = std::max(result.iterations, iteration + 1);
      if (unresolved <= operators.tolerance) break;
      if (control.require_convergence &&
          iteration + 1 == maximum_iterations)
        throw std::runtime_error("Neumann solve did not converge");
    }
  }
  result.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  populate_float32_efficiency_loss_upper_bound(operators, sources.count,
                                                result);
  return result;
}

void populate_float32_efficiency_loss_upper_bound(
    const OperatorSet& operators, std::uint64_t source_count,
    SolveResult& result) {
  if (result.losses.size() != source_count * operators.losses.cols)
    throw std::runtime_error(
        "cannot derive float32 efficiency-loss bounds from malformed losses");
  result.float32_efficiency_loss_upper_bound.assign(source_count, 0.0);
  const auto found =
      std::find(operators.loss_names.begin(), operators.loss_names.end(),
                "float32_intersection_miss");
  if (found == operators.loss_names.end()) return;
  const auto column =
      static_cast<std::uint64_t>(found - operators.loss_names.begin());
  for (std::uint64_t source = 0; source < source_count; ++source)
    result.float32_efficiency_loss_upper_bound[source] =
        result.losses[source * operators.losses.cols + column];
}

}  // namespace oos
