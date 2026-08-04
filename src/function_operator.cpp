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

double dot(const std::vector<double>& left,
           const std::vector<double>& right) {
  if (left.size() != right.size())
    throw std::runtime_error("function operator inner-product shape mismatch");
  return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

bool finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

}  // namespace

FunctionOperator::FunctionOperator(const std::filesystem::path& library,
                                   const std::string& config_json,
                                   double energy_eV) {
  library_handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library_handle_) throw std::runtime_error(dlerror());
  auto getter = reinterpret_cast<oos_get_function_operator_v2_fn>(
      dlsym(library_handle_, "oos_get_function_operator_v2"));
  if (!getter) {
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error(
        "functional surface does not export oos_get_function_operator_v2");
  }
  api_ = getter();
  if (!api_ || api_->abi_version != OOS_FUNCTION_OPERATOR_ABI_V2 ||
      !api_->create || !api_->destroy || !api_->apply_cpu ||
      !api_->apply_adjoint_cpu) {
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error("function operator ABI V2 mismatch");
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
  const bool valid_cuda =
      !descriptor_.supports_cuda ||
      (api_->prepare_cuda && api_->apply_cuda && api_->apply_adjoint_cuda);
  if (descriptor_.abi_version != OOS_FUNCTION_OPERATOR_ABI_V2 ||
      descriptor_.input_state_count == 0 ||
      descriptor_.retained_state_count != descriptor_.input_state_count ||
      descriptor_.contraction_bound < 0.0 ||
      descriptor_.contraction_bound >= 1.0 ||
      !descriptor_.supports_cpu || !valid_cuda) {
    api_->destroy(instance_);
    instance_ = nullptr;
    dlclose(library_handle_);
    library_handle_ = nullptr;
    throw std::runtime_error("function operator descriptor V2 is invalid");
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
  if (api_->apply_cpu(instance_, batch, input.data(), result.retained.data(),
                      result.egress.data(), result.losses.data()) != 0)
    throw std::runtime_error("function operator CPU application failed");
  return result;
}

std::vector<double> FunctionOperator::apply_adjoint_cpu(
    std::uint64_t batch, const std::vector<double>& retained_adjoint,
    const std::vector<double>& egress_adjoint,
    const std::vector<double>& losses_adjoint) const {
  const auto& shape = descriptor_;
  if (retained_adjoint.size() != batch * shape.retained_state_count ||
      egress_adjoint.size() != batch * shape.egress_count ||
      losses_adjoint.size() != batch * shape.loss_count)
    throw std::runtime_error("function operator adjoint shape mismatch");
  std::vector<double> result(batch * shape.input_state_count, 0.0);
  if (api_->apply_adjoint_cpu(
          instance_, batch, retained_adjoint.data(), egress_adjoint.data(),
          losses_adjoint.data(), result.data()) != 0)
    throw std::runtime_error("function operator CPU adjoint failed");
  return result;
}

void FunctionOperator::prepare_cuda(std::int32_t device) const {
  if (!descriptor_.supports_cuda)
    throw std::runtime_error(
        "function operator does not provide a CUDA implementation");
  if (api_->prepare_cuda(instance_, device) != 0)
    throw std::runtime_error("function operator CUDA preparation failed");
}

void FunctionOperator::apply_cuda(
    std::uint64_t batch, const double* device_input,
    double* device_retained, double* device_egress,
    double* device_losses, void* cuda_stream) const {
  if (!descriptor_.supports_cuda)
    throw std::runtime_error(
        "function operator does not provide a CUDA implementation");
  if (api_->apply_cuda(instance_, batch, device_input, device_retained,
                       device_egress, device_losses, cuda_stream) != 0)
    throw std::runtime_error("function operator CUDA application failed");
}

void FunctionOperator::apply_adjoint_cuda(
    std::uint64_t batch, const double* device_retained_adjoint,
    const double* device_egress_adjoint,
    const double* device_losses_adjoint, double* device_input_adjoint,
    void* cuda_stream) const {
  if (!descriptor_.supports_cuda)
    throw std::runtime_error(
        "function operator does not provide a CUDA implementation");
  if (api_->apply_adjoint_cuda(
          instance_, batch, device_retained_adjoint, device_egress_adjoint,
          device_losses_adjoint, device_input_adjoint, cuda_stream) != 0)
    throw std::runtime_error("function operator CUDA adjoint failed");
}

void validate_function_operator(FunctionOperator& function,
                                double tolerance) {
  const auto& descriptor = function.descriptor();
  constexpr std::uint64_t batch = 3;
  std::vector<double> input(batch * descriptor.input_state_count);
  std::vector<double> retained_seed(batch * descriptor.retained_state_count);
  std::vector<double> egress_seed(batch * descriptor.egress_count);
  std::vector<double> loss_seed(batch * descriptor.loss_count);
  const auto fill_probe = [](std::vector<double>& values,
                             std::uint64_t multiplier) {
    for (std::uint64_t index = 0; index < values.size(); ++index)
      values[index] =
          (static_cast<double>((index * multiplier + 17) % 251) - 125.0) /
          127.0;
  };
  fill_probe(input, 37);
  fill_probe(retained_seed, 41);
  fill_probe(egress_seed, 43);
  fill_probe(loss_seed, 47);
  const auto forward = function.apply_cpu(batch, input);
  const auto adjoint = function.apply_adjoint_cpu(
      batch, retained_seed, egress_seed, loss_seed);
  if (!finite(forward.retained) || !finite(forward.egress) ||
      !finite(forward.losses) || !finite(adjoint))
    throw std::runtime_error("function operator produced non-finite values");
  const double left = dot(forward.retained, retained_seed) +
                      dot(forward.egress, egress_seed) +
                      dot(forward.losses, loss_seed);
  const double right = dot(input, adjoint);
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  if (std::abs(left - right) > tolerance * scale)
    throw std::runtime_error(
        "function operator forward/adjoint inner-product check failed");
}

}  // namespace oos
