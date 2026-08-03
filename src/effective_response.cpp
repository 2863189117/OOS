#include "oos/effective_response.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace oos {

void EffectiveResponse::validate() const {
  if (states == 0 || channels == 0 || cycles == 0 ||
      state_to_detection.size() != states * channels ||
      state_to_losses.size() != states * losses ||
      state_unresolved.size() != states || channel_ids.size() != channels ||
      loss_names.size() != losses || operator_cache_key_sha256.empty()) {
    throw std::runtime_error("effective response dimensions are invalid");
  }
  const auto valid = [](const auto& values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value) && value >= 0.0;
    });
  };
  if (!valid(state_to_detection) || !valid(state_to_losses) ||
      !valid(state_unresolved))
    throw std::runtime_error(
        "effective response entries must be finite and nonnegative");
  for (std::uint64_t state = 0; state < states; ++state) {
    double total = state_unresolved[state];
    total += std::accumulate(
        state_to_detection.begin() + state * channels,
        state_to_detection.begin() + (state + 1) * channels, 0.0);
    total += std::accumulate(state_to_losses.begin() + state * losses,
                             state_to_losses.begin() + (state + 1) * losses,
                             0.0);
    if (total > 1.0 + 1.0e-8)
      throw std::runtime_error("effective response row creates energy");
  }
}

EffectiveResponse build_effective_response_cpu(const OperatorSet& operators,
                                               std::uint32_t cycles,
                                               std::uint64_t batch_size) {
  operators.validate();
  if (cycles == 0 || batch_size == 0)
    throw std::runtime_error(
        "effective response cycles and batch size must be positive");
  EffectiveResponse response;
  response.states = operators.transition.rows;
  response.channels = operators.detection.cols;
  response.losses = operators.losses.cols;
  response.cycles = cycles;
  response.state_to_detection.assign(response.states * response.channels, 0.0);
  response.state_to_losses.assign(response.states * response.losses, 0.0);
  response.state_unresolved.assign(response.states, 0.0);
  response.channel_ids = operators.channel_ids;
  response.loss_names = operators.loss_names;
  response.operator_cache_key_sha256 = operators.cache_key_sha256;
  response.code_commit = operators.code_commit;

  for (std::uint64_t start = 0; start < response.states;
       start += batch_size) {
    const auto count = std::min(batch_size, response.states - start);
    SourceBatch sources;
    sources.count = count;
    sources.initial_states.assign(count * response.states, 0.0);
    sources.direct_detection.assign(count * response.channels, 0.0);
    sources.direct_losses.assign(count * response.losses, 0.0);
    sources.source_integration_l1_error_estimate.assign(count, 0.0);
    for (std::uint64_t row = 0; row < count; ++row)
      sources.initial_states[row * response.states + start + row] = 1.0;
    const auto solved = Solver::solve_cpu(
        operators, sources, SolveControl{cycles, false});
    for (std::uint64_t row = 0; row < count; ++row) {
      std::copy_n(solved.efficiency.begin() + row * response.channels,
                  response.channels,
                  response.state_to_detection.begin() +
                      (start + row) * response.channels);
      std::copy_n(solved.losses.begin() + row * response.losses,
                  response.losses,
                  response.state_to_losses.begin() +
                      (start + row) * response.losses);
      response.state_unresolved[start + row] = solved.unresolved[row];
    }
  }
  response.validate();
  return response;
}

SolveResult apply_effective_response_cpu(const EffectiveResponse& response,
                                         const SourceBatch& sources) {
  const auto started = std::chrono::steady_clock::now();
  response.validate();
  if (sources.initial_states.size() != sources.count * response.states ||
      sources.direct_detection.size() != sources.count * response.channels ||
      sources.direct_losses.size() != sources.count * response.losses)
    throw std::runtime_error("source batch shape mismatch");

  SolveResult result;
  result.backend = "cpu-precomputed";
  result.hardware = "native-cpu dense bounded response";
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

#pragma omp parallel for schedule(static)
  for (std::int64_t source = 0;
       source < static_cast<std::int64_t>(sources.count); ++source) {
    const auto source_offset =
        static_cast<std::uint64_t>(source) * response.states;
    const auto detection_offset =
        static_cast<std::uint64_t>(source) * response.channels;
    const auto loss_offset =
        static_cast<std::uint64_t>(source) * response.losses;
    double unresolved = 0.0;
    for (std::uint64_t state = 0; state < response.states; ++state) {
      const double weight = sources.initial_states[source_offset + state];
      if (weight == 0.0) continue;
      const auto state_detection = state * response.channels;
      for (std::uint64_t channel = 0; channel < response.channels; ++channel)
        result.efficiency[detection_offset + channel] +=
            weight * response.state_to_detection[state_detection + channel];
      const auto state_loss = state * response.losses;
      for (std::uint64_t loss = 0; loss < response.losses; ++loss)
        result.losses[loss_offset + loss] +=
            weight * response.state_to_losses[state_loss + loss];
      unresolved += weight * response.state_unresolved[state];
    }
    result.unresolved[source] = unresolved;
    result.input_weight[source] =
        std::accumulate(
            sources.initial_states.begin() + source_offset,
            sources.initial_states.begin() + source_offset + response.states,
            0.0) +
        std::accumulate(
            sources.direct_detection.begin() + detection_offset,
            sources.direct_detection.begin() + detection_offset +
                response.channels,
            0.0) +
        std::accumulate(sources.direct_losses.begin() + loss_offset,
                        sources.direct_losses.begin() + loss_offset +
                            response.losses,
                        0.0);
  }
  result.iterations = response.cycles;
  result.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  OperatorSet audit;
  audit.loss_names = response.loss_names;
  audit.losses.cols = response.losses;
  populate_float32_efficiency_loss_upper_bound(audit, sources.count, result);
  return result;
}

}  // namespace oos
