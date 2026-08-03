#include "oos/function_operator.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace oos {
namespace {

oos_string_view_v1 view(const std::string& value) {
  return {value.data(), value.size()};
}

std::string text(oos_string_view_v1 value) {
  return std::string(value.data ? value.data : "", value.size);
}

double sum_row(const std::vector<double>& values, std::uint64_t row,
               std::uint64_t columns) {
  return std::accumulate(values.begin() + row * columns,
                         values.begin() + (row + 1) * columns, 0.0);
}

}  // namespace

FunctionOperator::FunctionOperator(const std::filesystem::path& library,
                                   const std::string& config_json,
                                   double energy_eV) {
  library_handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library_handle_) throw std::runtime_error(dlerror());
  auto getter = reinterpret_cast<oos_get_function_operator_v1_fn>(
      dlsym(library_handle_, "oos_get_function_operator_v1"));
  if (!getter) {
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error(
        "functional surface does not export oos_get_function_operator_v1");
  }
  api_ = getter();
  if (!api_ || api_->abi_version != OOS_FUNCTION_OPERATOR_ABI_V1 ||
      !api_->create || !api_->destroy || !api_->apply_cpu) {
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error("function operator ABI mismatch");
  }
  if (api_->validate) {
    const auto validation = api_->validate(view(config_json), energy_eV);
    if (validation.status != 0) {
      const auto message = text(validation.message);
      dlclose(library_handle_);
      library_handle_ = nullptr;
      throw std::runtime_error(message);
    }
  }
  if (api_->create(view(config_json), energy_eV, &instance_, &descriptor_) !=
          0 ||
      !instance_) {
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error("function operator instance creation failed");
  }
  if (descriptor_.abi_version != OOS_FUNCTION_OPERATOR_ABI_V1 ||
      descriptor_.input_state_count == 0 ||
      descriptor_.retained_state_count !=
          descriptor_.input_state_count ||
      descriptor_.contraction_bound < 0.0 ||
      descriptor_.contraction_bound >= 1.0 ||
      !descriptor_.supports_cpu) {
    api_->destroy(instance_);
    instance_ = nullptr;
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error("function operator descriptor is invalid");
  }
}

FunctionOperator::~FunctionOperator() {
  if (instance_ && api_) api_->destroy(instance_);
  if (library_handle_) dlclose(library_handle_);
}

std::string FunctionOperator::name() const { return text(api_->name); }
std::string FunctionOperator::version() const { return text(api_->version); }

FunctionApplyResult FunctionOperator::apply_cpu(
    std::uint64_t batch, const std::vector<double>& input) const {
  const auto& shape = descriptor_;
  if (input.size() != batch * shape.input_state_count)
    throw std::runtime_error("function operator input shape mismatch");
  FunctionApplyResult result;
  result.batch = batch;
  result.retained.assign(batch * shape.retained_state_count, 0.0);
  result.egress.assign(batch * shape.egress_count, 0.0);
  result.losses.assign(batch * shape.loss_count, 0.0);
  result.audit.resize(batch);
  if (api_->apply_cpu(instance_, batch, input.data(), result.retained.data(),
                      result.egress.data(), result.losses.data(),
                      result.audit.data()) != 0)
    throw std::runtime_error("function operator CPU application failed");
  return result;
}

void FunctionOperator::prepare_cuda(std::int32_t device) const {
  if (!descriptor_.supports_cuda || !api_->prepare_cuda || !api_->apply_cuda)
    throw std::runtime_error(
        "function operator does not provide a CUDA implementation");
  if (api_->prepare_cuda(instance_, device) != 0)
    throw std::runtime_error("function operator CUDA preparation failed");
}

void FunctionOperator::apply_cuda(
    std::uint64_t batch, const double* device_input,
    double* device_retained, double* device_egress,
    double* device_losses, void* cuda_stream) const {
  if (!descriptor_.supports_cuda || !api_->apply_cuda)
    throw std::runtime_error(
        "function operator does not provide a CUDA implementation");
  if (api_->apply_cuda(instance_, batch, device_input, device_retained,
                       device_egress, device_losses, cuda_stream,
                       nullptr) != 0)
    throw std::runtime_error("function operator CUDA application failed");
}

void validate_function_operator(FunctionOperator& function,
                                double tolerance) {
  const auto& descriptor = function.descriptor();
  const auto states = descriptor.input_state_count;
  const std::uint64_t probes = std::min<std::uint64_t>(states, 8);
  std::vector<double> input((2 * probes + 3) * states, 0.0);
  for (std::uint64_t probe = 0; probe < probes; ++probe) {
    input[probe * states + probe] = 1.0;
    input[(probes + probe) * states + probe] = 0.375;
  }
  for (std::uint64_t state = 0; state < states; ++state) {
    const double value =
        static_cast<double>((state * 17 + 11) % 101 + 1) / 102.0;
    input[(2 * probes) * states + state] = value;
    input[(2 * probes + 1) * states + state] = 0.25 * value;
    input[(2 * probes + 2) * states + state] = 1.25 * value;
  }
  const auto result = function.apply_cpu(2 * probes + 3, input);
  const auto check_values = [](const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value) && value >= 0.0;
    });
  };
  if (!check_values(result.retained) || !check_values(result.egress) ||
      !check_values(result.losses))
    throw std::runtime_error(
        "function operator violates non-negativity or finiteness");
  for (std::uint64_t row = 0; row < result.batch; ++row) {
    const double input_weight = sum_row(input, row, states);
    const double output_weight =
        sum_row(result.retained, row, descriptor.retained_state_count) +
        sum_row(result.egress, row, descriptor.egress_count) +
        sum_row(result.losses, row, descriptor.loss_count);
    if (std::abs(output_weight - input_weight) >
        tolerance * std::max(1.0, input_weight))
      throw std::runtime_error("function operator does not conserve weight");
    if (std::abs(result.audit[row].closure_error) >
        tolerance * std::max(1.0, input_weight))
      throw std::runtime_error(
          "function operator audit reports non-closure");
  }
  const auto compare_scaled =
      [&](const std::vector<double>& values, std::uint64_t columns,
          std::uint64_t full_row, std::uint64_t scaled_row, double scale) {
        for (std::uint64_t column = 0; column < columns; ++column) {
          const double expected = scale * values[full_row * columns + column];
          const double actual = values[scaled_row * columns + column];
          if (std::abs(actual - expected) >
              tolerance * std::max({1.0, std::abs(expected), std::abs(actual)}))
            throw std::runtime_error(
                "function operator violates linear homogeneity");
        }
      };
  for (std::uint64_t probe = 0; probe < probes; ++probe) {
    compare_scaled(result.retained, descriptor.retained_state_count, probe,
                   probes + probe, 0.375);
    compare_scaled(result.egress, descriptor.egress_count, probe,
                   probes + probe, 0.375);
    compare_scaled(result.losses, descriptor.loss_count, probe,
                   probes + probe, 0.375);
  }
  compare_scaled(result.retained, descriptor.retained_state_count,
                 2 * probes, 2 * probes + 1, 0.25);
  compare_scaled(result.egress, descriptor.egress_count, 2 * probes,
                 2 * probes + 1, 0.25);
  compare_scaled(result.losses, descriptor.loss_count, 2 * probes,
                 2 * probes + 1, 0.25);
  compare_scaled(result.retained, descriptor.retained_state_count,
                 2 * probes, 2 * probes + 2, 1.25);
  compare_scaled(result.egress, descriptor.egress_count, 2 * probes,
                 2 * probes + 2, 1.25);
  compare_scaled(result.losses, descriptor.loss_count, 2 * probes,
                 2 * probes + 2, 1.25);
}

}  // namespace oos
