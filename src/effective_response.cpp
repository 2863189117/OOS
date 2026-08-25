#include "oos/effective_response.hpp"

#include "oos/function_operator.hpp"
#include "oos/hash.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace oos {
namespace {

std::vector<double> adjoint_multiply_batch(const CsrMatrix& matrix,
                                           const std::vector<double>& input,
                                           std::uint64_t batch) {
  if (input.size() != batch * matrix.cols)
    throw std::runtime_error("batched adjoint matrix shape mismatch");
  std::vector<double> output(batch * matrix.rows, 0.0);
#pragma omp parallel for schedule(static)
  for (std::int64_t signed_batch = 0;
       signed_batch < static_cast<std::int64_t>(batch); ++signed_batch) {
    const auto batch_row = static_cast<std::uint64_t>(signed_batch);
    for (std::uint64_t row = 0; row < matrix.rows; ++row) {
      double value = 0.0;
      for (std::uint64_t entry = matrix.indptr[row];
           entry < matrix.indptr[row + 1]; ++entry)
        value += matrix.data[entry] *
                 input[batch_row * matrix.cols + matrix.indices[entry]];
      output[batch_row * matrix.rows + row] = value;
    }
  }
  return output;
}

void add_in_place(std::vector<double>& target,
                  const std::vector<double>& values) {
  if (target.size() != values.size())
    throw std::runtime_error("batched accumulation shape mismatch");
#pragma omp parallel for schedule(static)
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(target.size()); ++index)
    target[static_cast<std::size_t>(index)] +=
        values[static_cast<std::size_t>(index)];
}

std::vector<std::unique_ptr<FunctionOperator>> load_functions(
    const OperatorSet& operators) {
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
  return functions;
}

std::vector<double> apply_adjoint_cycle(
    const OperatorSet& operators,
    const std::vector<std::unique_ptr<FunctionOperator>>& functions,
    const std::vector<double>& next_state,
    const std::vector<double>& detection_seed,
    const std::vector<double>& loss_seed, std::uint64_t batch) {
  const auto states = operators.transition.rows;
  auto current =
      adjoint_multiply_batch(operators.transition, next_state, batch);
  add_in_place(
      current,
      adjoint_multiply_batch(operators.detection, detection_seed, batch));
  add_in_place(current,
               adjoint_multiply_batch(operators.losses, loss_seed, batch));
  for (std::size_t block_index = 0;
       block_index < operators.function_blocks.size(); ++block_index) {
    const auto& block = operators.function_blocks[block_index];
    std::vector<double> retained_adjoint(batch * block.state_count);
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_batch = 0;
         signed_batch < static_cast<std::int64_t>(batch); ++signed_batch) {
      const auto batch_row = static_cast<std::uint64_t>(signed_batch);
      std::copy_n(next_state.begin() + batch_row * states +
                      block.state_offset,
                  block.state_count,
                  retained_adjoint.begin() + batch_row * block.state_count);
    }
    auto egress_adjoint = adjoint_multiply_batch(
        block.egress_to_transition, next_state, batch);
    add_in_place(egress_adjoint,
                 adjoint_multiply_batch(block.egress_to_detection,
                                        detection_seed, batch));
    add_in_place(egress_adjoint,
                 adjoint_multiply_batch(block.egress_to_losses, loss_seed,
                                        batch));
    std::vector<double> intrinsic_loss_adjoint(
        batch * block.intrinsic_loss_count, 0.0);
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_batch = 0;
         signed_batch < static_cast<std::int64_t>(batch); ++signed_batch) {
      const auto batch_row = static_cast<std::uint64_t>(signed_batch);
      for (std::uint64_t loss = 0; loss < block.intrinsic_loss_count; ++loss)
        intrinsic_loss_adjoint[batch_row * block.intrinsic_loss_count + loss] =
            loss_seed[batch_row * operators.losses.cols +
                      block.intrinsic_loss_columns[loss]];
    }
    const auto input_adjoint = functions[block_index]->apply_adjoint_cpu(
        batch, retained_adjoint, egress_adjoint, intrinsic_loss_adjoint);
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_batch = 0;
         signed_batch < static_cast<std::int64_t>(batch); ++signed_batch) {
      const auto batch_row = static_cast<std::uint64_t>(signed_batch);
      for (std::uint64_t state = 0; state < block.state_count; ++state)
        current[batch_row * states + block.state_offset + state] +=
            input_adjoint[batch_row * block.state_count + state];
    }
  }
  return current;
}

}  // namespace

std::string effective_response_fingerprint(
    const EffectiveResponse& response) {
  std::ostringstream material;
  material << "oos.effective-response.semantic.v1\n"
           << response.operator_cache_key_sha256 << '\n'
           << response.construction_method << '\n'
           << response.cycles << '\n'
           << std::setprecision(17) << response.operator_tolerance << '\n'
           << response.states << '\n'
           << response.channels << '\n'
           << response.losses << '\n'
           << response.code_commit;
  return sha256_string(material.str());
}

void EffectiveResponse::validate() const {
  if (states == 0 || channels == 0 || cycles == 0 || build_batch_size == 0 ||
      !std::isfinite(operator_tolerance) || operator_tolerance <= 0.0 ||
      construction_method != "adjoint_linear" ||
      state_to_detection.size() != states * channels ||
      state_to_losses.size() != states * losses ||
      state_unresolved.size() != states || channel_ids.size() != channels ||
      loss_names.size() != losses || operator_cache_key_sha256.empty())
    throw std::runtime_error("effective response dimensions are invalid");
  const auto finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (!finite(state_to_detection) || !finite(state_to_losses) ||
      !finite(state_unresolved))
    throw std::runtime_error(
        "adjoint effective response contains non-finite entries");
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
  response.build_batch_size = batch_size;
  response.operator_tolerance = operators.tolerance;
  response.state_to_detection.assign(response.states * response.channels,
                                     0.0);
  response.state_to_losses.assign(response.states * response.losses, 0.0);
  response.state_unresolved.assign(response.states, 0.0);
  response.channel_ids = operators.channel_ids;
  response.loss_names = operators.loss_names;
  response.operator_cache_key_sha256 = operators.cache_key_sha256;
  response.code_commit = operators.code_commit;
  const auto functions = load_functions(operators);

  const auto build_terminal =
      [&](std::uint64_t terminal_count, bool detection,
          std::vector<double>& output) {
        for (std::uint64_t start = 0; start < terminal_count;
             start += batch_size) {
          const auto count = std::min(batch_size, terminal_count - start);
          std::vector<double> detection_seed(count * response.channels, 0.0);
          std::vector<double> loss_seed(count * response.losses, 0.0);
          auto& seed = detection ? detection_seed : loss_seed;
          const auto columns = detection ? response.channels : response.losses;
          for (std::uint64_t row = 0; row < count; ++row)
            seed[row * columns + start + row] = 1.0;
          std::vector<double> state_value(count * response.states, 0.0);
          for (std::uint32_t cycle = 0; cycle < cycles; ++cycle)
            state_value = apply_adjoint_cycle(
                operators, functions, state_value, detection_seed, loss_seed,
                count);
#pragma omp parallel for schedule(static)
          for (std::int64_t signed_state = 0;
               signed_state < static_cast<std::int64_t>(response.states);
               ++signed_state) {
            const auto state = static_cast<std::uint64_t>(signed_state);
            for (std::uint64_t row = 0; row < count; ++row)
              output[state * terminal_count + start + row] =
                  state_value[row * response.states + state];
          }
        }
      };
  build_terminal(response.channels, true, response.state_to_detection);
  build_terminal(response.losses, false, response.state_to_losses);

  std::vector<double> unresolved_state(response.states, 1.0);
  const std::vector<double> no_detection(response.channels, 0.0);
  const std::vector<double> no_loss(response.losses, 0.0);
  for (std::uint32_t cycle = 0; cycle < cycles; ++cycle)
    unresolved_state = apply_adjoint_cycle(
        operators, functions, unresolved_state, no_detection, no_loss, 1);
  response.state_unresolved = std::move(unresolved_state);
  response.fingerprint_sha256 = effective_response_fingerprint(response);
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
  result.backend = "cpu-precomputed-adjoint";
  result.hardware = "native-cpu dense bounded adjoint response";
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
