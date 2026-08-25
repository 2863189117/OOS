#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "oos/plugin.h"

struct LXeFunctionInstance {
  std::uint64_t nr{};
  std::uint64_t np{};
  std::uint64_t nm{};
  std::uint64_t nd{};
  std::uint64_t orders{};
  std::uint64_t surface_radial{};
  std::uint64_t surface_phi{};
  std::uint64_t angular{};
  std::vector<std::complex<double>> coefficients;
  std::vector<double> expected_return;
  std::vector<double> surface_ring_area;
  std::vector<double> angular_weight;
  void* cuda_state{};
};

#ifdef OOS_LXE_HAS_CUDA
int prepare_lxe_function_cuda(LXeFunctionInstance*, std::int32_t device);
int apply_lxe_function_cuda(
    LXeFunctionInstance*, std::uint64_t batch, const double* device_input,
    double* device_retained, double* device_egress, double* device_losses,
    void* cuda_stream);
int apply_lxe_function_adjoint_cuda(
    LXeFunctionInstance*, std::uint64_t batch,
    const double* device_retained_adjoint,
    const double* device_egress_adjoint,
    const double* device_losses_adjoint, double* device_input_adjoint,
    void* cuda_stream);
void destroy_lxe_function_cuda(LXeFunctionInstance*);
#else
inline int prepare_lxe_function_cuda(LXeFunctionInstance*, std::int32_t) {
  return 1;
}
inline int apply_lxe_function_cuda(
    LXeFunctionInstance*, std::uint64_t, const double*, double*, double*,
    double*, void*) {
  return 1;
}
inline int apply_lxe_function_adjoint_cuda(
    LXeFunctionInstance*, std::uint64_t, const double*, const double*,
    const double*, double*, void*) {
  return 1;
}
inline void destroy_lxe_function_cuda(LXeFunctionInstance*) {}
#endif
