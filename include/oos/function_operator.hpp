#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "oos/plugin.h"

namespace oos {

struct FunctionApplyResult {
  std::uint64_t batch{};
  std::vector<double> retained;
  std::vector<double> egress;
  std::vector<double> losses;
};

class FunctionOperator {
 public:
  FunctionOperator(const std::filesystem::path& library,
                   const std::string& config_json, double energy_eV);
  ~FunctionOperator();
  FunctionOperator(const FunctionOperator&) = delete;
  FunctionOperator& operator=(const FunctionOperator&) = delete;
  FunctionOperator(FunctionOperator&&) = delete;
  FunctionOperator& operator=(FunctionOperator&&) = delete;

  const oos_function_operator_descriptor_v2& descriptor() const {
    return descriptor_;
  }
  std::string name() const;
  std::string version() const;
  FunctionApplyResult apply_cpu(std::uint64_t batch,
                                const std::vector<double>& input) const;
  std::vector<double> apply_adjoint_cpu(
      std::uint64_t batch, const std::vector<double>& retained_adjoint,
      const std::vector<double>& egress_adjoint,
      const std::vector<double>& losses_adjoint) const;
  void prepare_cuda(std::int32_t device) const;
  void apply_cuda(std::uint64_t batch, const double* device_input,
                  double* device_retained, double* device_egress,
                  double* device_losses, void* cuda_stream = nullptr) const;
  void apply_adjoint_cuda(
      std::uint64_t batch, const double* device_retained_adjoint,
      const double* device_egress_adjoint,
      const double* device_losses_adjoint, double* device_input_adjoint,
      void* cuda_stream = nullptr) const;

 private:
  void* library_handle_{};
  const oos_function_operator_v2* api_{};
  void* instance_{};
  oos_function_operator_descriptor_v2 descriptor_{};
};

void validate_function_operator(FunctionOperator& function,
                                double tolerance = 1.0e-10);

}  // namespace oos
