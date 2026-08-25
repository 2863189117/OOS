#include "lxe_function.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct CudaState {
  int device{};
  std::uint64_t capacity{};
  cuDoubleComplex* coefficients{};
  double* expected_return{};
  double* surface_ring_area{};
  double* angular_weight{};
  cuDoubleComplex* phase_modes{};
  cuDoubleComplex* modal_surface{};
  double* surface_weight{};
  double* input_total{};
  double* expected_total{};
};

bool okay(cudaError_t status) { return status == cudaSuccess; }

void release_scratch(CudaState& state) {
  cudaFree(state.phase_modes);
  cudaFree(state.modal_surface);
  cudaFree(state.surface_weight);
  cudaFree(state.input_total);
  cudaFree(state.expected_total);
  state.phase_modes = nullptr;
  state.modal_surface = nullptr;
  state.surface_weight = nullptr;
  state.input_total = nullptr;
  state.expected_total = nullptr;
  state.capacity = 0;
}

bool ensure_capacity(const LXeFunctionInstance& function, CudaState& state,
                     std::uint64_t batch) {
  if (batch <= state.capacity) return true;
  release_scratch(state);
  const auto phase_mode_count =
      batch * function.nr * function.nm * function.nd * function.orders;
  const auto modal_count =
      batch * function.orders * function.surface_radial;
  const auto surface_count =
      batch * function.surface_radial * function.surface_phi;
  if (!okay(cudaMalloc(&state.phase_modes,
                       phase_mode_count * sizeof(cuDoubleComplex))) ||
      !okay(cudaMalloc(&state.modal_surface,
                       modal_count * sizeof(cuDoubleComplex))) ||
      !okay(cudaMalloc(&state.surface_weight,
                       surface_count * sizeof(double))) ||
      !okay(cudaMalloc(&state.input_total, batch * sizeof(double))) ||
      !okay(cudaMalloc(&state.expected_total, batch * sizeof(double)))) {
    release_scratch(state);
    return false;
  }
  state.capacity = batch;
  return true;
}

__global__ void phase_modes_kernel(
    const double* input, std::uint64_t batch, std::uint64_t nr,
    std::uint64_t np, std::uint64_t nm, std::uint64_t nd,
    std::uint64_t orders, cuDoubleComplex* output) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * nr * nm * nd * orders;
  if (linear >= count) return;
  auto value = linear;
  const auto order = value % orders;
  value /= orders;
  const auto direction_phi = value % nd;
  value /= nd;
  const auto mu = value % nm;
  value /= nm;
  const auto radial = value % nr;
  const auto row = value / nr;
  constexpr double two_pi =
      6.283185307179586476925286766559005768;
  double real = 0.0;
  double imaginary = 0.0;
  for (std::uint64_t position_phi = 0; position_phi < np; ++position_phi) {
    const auto state =
        (((radial * np + position_phi) * nm + mu) * nd + direction_phi);
    const double weight = input[row * (nr * np * nm * nd) + state];
    const double angle =
        -static_cast<double>(order) * two_pi *
        static_cast<double>(position_phi) / static_cast<double>(np);
    real += weight * cos(angle);
    imaginary += weight * sin(angle);
  }
  output[linear] = make_cuDoubleComplex(real, imaginary);
}

__global__ void modal_surface_kernel(
    const cuDoubleComplex* phase_modes,
    const cuDoubleComplex* coefficients, std::uint64_t batch,
    std::uint64_t nr, std::uint64_t nm, std::uint64_t nd,
    std::uint64_t orders, std::uint64_t surface_radial,
    cuDoubleComplex* output) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * orders * surface_radial;
  if (linear >= count) return;
  auto value = linear;
  const auto surface = value % surface_radial;
  value /= surface_radial;
  const auto order = value % orders;
  const auto row = value / orders;
  cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (std::uint64_t radial = 0; radial < nr; ++radial)
    for (std::uint64_t mu = 0; mu < nm; ++mu)
      for (std::uint64_t direction_phi = 0; direction_phi < nd;
           ++direction_phi) {
        const auto phase_index =
            ((((row * nr + radial) * nm + mu) * nd + direction_phi) *
                 orders +
             order);
        const auto coefficient_index =
            ((((direction_phi * nr + radial) * nm + mu) * orders + order) *
                 surface_radial +
             surface);
        result = cuCadd(
            result,
            cuCmul(phase_modes[phase_index],
                   coefficients[coefficient_index]));
      }
  output[linear] = result;
}

__global__ void input_totals_kernel(
    const double* input, const double* expected_return,
    std::uint64_t batch, std::uint64_t nr, std::uint64_t np,
    std::uint64_t nm, std::uint64_t nd, double* input_total,
    double* expected_total) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= batch) return;
  double local_input = 0.0;
  double local_expected = 0.0;
  const auto states = nr * np * nm * nd;
  for (std::uint64_t state = threadIdx.x; state < states;
       state += blockDim.x) {
    auto value = state;
    const auto direction_phi = value % nd;
    value /= nd;
    const auto mu = value % nm;
    value /= nm;
    value /= np;
    const auto radial = value;
    const double weight = input[row * states + state];
    local_input += weight;
    local_expected +=
        weight * expected_return[
                     (direction_phi * nr + radial) * nm + mu];
  }
  __shared__ double input_partial[256];
  __shared__ double expected_partial[256];
  input_partial[threadIdx.x] = local_input;
  expected_partial[threadIdx.x] = local_expected;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      input_partial[threadIdx.x] += input_partial[threadIdx.x + stride];
      expected_partial[threadIdx.x] +=
          expected_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    input_total[row] = input_partial[0];
    expected_total[row] = expected_partial[0];
  }
}

__global__ void surface_weights_kernel(
    const cuDoubleComplex* modal_surface, const double* ring_area,
    std::uint64_t batch, std::uint64_t orders,
    std::uint64_t surface_radial, std::uint64_t surface_phi,
    double* surface_weight) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * surface_radial * surface_phi;
  if (linear >= count) return;
  auto value = linear;
  const auto phi_index = value % surface_phi;
  value /= surface_phi;
  const auto surface = value % surface_radial;
  const auto row = value / surface_radial;
  constexpr double two_pi =
      6.283185307179586476925286766559005768;
  const double phi =
      two_pi * (static_cast<double>(phi_index) + 0.5) /
      static_cast<double>(surface_phi);
  cuDoubleComplex density = make_cuDoubleComplex(0.0, 0.0);
  for (std::uint64_t order = 0; order < orders; ++order) {
    const auto mode =
        modal_surface[(row * orders + order) * surface_radial + surface];
    const auto basis =
        make_cuDoubleComplex(cos(order * phi), sin(order * phi));
    density = cuCadd(density, cuCmul(mode, basis));
  }
  surface_weight[linear] =
      cuCreal(density) * ring_area[surface] /
      static_cast<double>(surface_phi);
}

__global__ void egress_kernel(
    const double* surface_weight, const double* angular_weight,
    std::uint64_t batch, std::uint64_t surface_count,
    std::uint64_t angular, double* egress) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * surface_count * angular;
  if (linear >= count) return;
  const auto angle = linear % angular;
  const auto surface = (linear / angular) % surface_count;
  const auto row = linear / (surface_count * angular);
  egress[linear] =
      surface_weight[row * surface_count + surface] * angular_weight[angle];
}

__global__ void losses_kernel(const double* input_total,
                              const double* expected_total,
                              std::uint64_t batch, double* losses) {
  const auto row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < batch)
    losses[row] = input_total[row] - expected_total[row];
}

__global__ void egress_adjoint_surface_kernel(
    const double* egress_adjoint, const double* angular_weight,
    const double* ring_area, std::uint64_t batch,
    std::uint64_t surface_radial, std::uint64_t surface_phi,
    std::uint64_t angular, double* surface_adjoint) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto surface_count = surface_radial * surface_phi;
  const auto count = batch * surface_count;
  if (linear >= count) return;
  const auto surface_index = linear % surface_count;
  const auto surface = surface_index / surface_phi;
  const auto row = linear / surface_count;
  double value = 0.0;
  for (std::uint64_t angle = 0; angle < angular; ++angle)
    value += egress_adjoint[
                 (row * surface_count + surface_index) * angular + angle] *
             angular_weight[angle];
  surface_adjoint[linear] =
      value * ring_area[surface] / static_cast<double>(surface_phi);
}

__global__ void surface_modes_adjoint_kernel(
    const double* surface_adjoint, std::uint64_t batch,
    std::uint64_t orders, std::uint64_t surface_radial,
    std::uint64_t surface_phi, cuDoubleComplex* output) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * orders * surface_radial;
  if (linear >= count) return;
  auto value = linear;
  const auto surface = value % surface_radial;
  value /= surface_radial;
  const auto order = value % orders;
  const auto row = value / orders;
  constexpr double two_pi =
      6.283185307179586476925286766559005768;
  double real = 0.0;
  double imaginary = 0.0;
  for (std::uint64_t phi_index = 0; phi_index < surface_phi; ++phi_index) {
    const double seed = surface_adjoint[
        (row * surface_radial + surface) * surface_phi + phi_index];
    const double phi =
        two_pi * (static_cast<double>(phi_index) + 0.5) /
        static_cast<double>(surface_phi);
    real += seed * cos(order * phi);
    imaginary += seed * sin(order * phi);
  }
  output[linear] = make_cuDoubleComplex(real, imaginary);
}

__global__ void phase_modes_adjoint_kernel(
    const cuDoubleComplex* surface_modes,
    const cuDoubleComplex* coefficients, std::uint64_t batch,
    std::uint64_t nr, std::uint64_t nm, std::uint64_t nd,
    std::uint64_t orders, std::uint64_t surface_radial,
    cuDoubleComplex* output) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * nr * nm * nd * orders;
  if (linear >= count) return;
  auto value = linear;
  const auto order = value % orders;
  value /= orders;
  const auto direction_phi = value % nd;
  value /= nd;
  const auto mu = value % nm;
  value /= nm;
  const auto radial = value % nr;
  const auto row = value / nr;
  cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (std::uint64_t surface = 0; surface < surface_radial; ++surface) {
    const auto coefficient_index =
        ((((direction_phi * nr + radial) * nm + mu) * orders + order) *
             surface_radial +
         surface);
    result = cuCadd(
        result,
        cuCmul(coefficients[coefficient_index],
               surface_modes[(row * orders + order) * surface_radial +
                             surface]));
  }
  output[linear] = result;
}

__global__ void input_adjoint_kernel(
    const cuDoubleComplex* phase_adjoint, const double* expected_return,
    const double* losses_adjoint, std::uint64_t batch,
    std::uint64_t nr, std::uint64_t np, std::uint64_t nm,
    std::uint64_t nd, std::uint64_t orders, double* input_adjoint) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto states = nr * np * nm * nd;
  const auto count = batch * states;
  if (linear >= count) return;
  auto state = linear % states;
  const auto row = linear / states;
  const auto direction_phi = state % nd;
  state /= nd;
  const auto mu = state % nm;
  state /= nm;
  const auto position_phi = state % np;
  const auto radial = state / np;
  const auto expected_index =
      (direction_phi * nr + radial) * nm + mu;
  double result = losses_adjoint[row] *
                  (1.0 - expected_return[expected_index]);
  constexpr double two_pi =
      6.283185307179586476925286766559005768;
  const double phi = two_pi * static_cast<double>(position_phi) /
                     static_cast<double>(np);
  for (std::uint64_t order = 0; order < orders; ++order) {
    const auto mode =
        phase_adjoint[
            ((((row * nr + radial) * nm + mu) * nd + direction_phi) *
                 orders +
             order)];
    result += cuCreal(cuCmul(
        mode,
        make_cuDoubleComplex(cos(order * phi), -sin(order * phi))));
  }
  input_adjoint[linear] = result;
}

}  // namespace

int prepare_lxe_function_cuda(LXeFunctionInstance* function,
                              std::int32_t device) {
  if (!function) return 1;
  if (!okay(cudaSetDevice(device))) return 2;
  if (function->cuda_state) return 0;
  auto* state = new CudaState;
  state->device = device;
  std::vector<cuDoubleComplex> coefficient(function->coefficients.size());
  for (std::size_t index = 0; index < coefficient.size(); ++index)
    coefficient[index] =
        make_cuDoubleComplex(function->coefficients[index].real(),
                            function->coefficients[index].imag());
  if (!okay(cudaMalloc(&state->coefficients,
                       coefficient.size() * sizeof(cuDoubleComplex))) ||
      !okay(cudaMalloc(&state->expected_return,
                       function->expected_return.size() * sizeof(double))) ||
      !okay(cudaMalloc(&state->surface_ring_area,
                       function->surface_ring_area.size() * sizeof(double))) ||
      !okay(cudaMalloc(&state->angular_weight,
                       function->angular_weight.size() * sizeof(double))) ||
      !okay(cudaMemcpy(state->coefficients, coefficient.data(),
                       coefficient.size() * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice)) ||
      !okay(cudaMemcpy(state->expected_return,
                       function->expected_return.data(),
                       function->expected_return.size() * sizeof(double),
                       cudaMemcpyHostToDevice)) ||
      !okay(cudaMemcpy(state->surface_ring_area,
                       function->surface_ring_area.data(),
                       function->surface_ring_area.size() * sizeof(double),
                       cudaMemcpyHostToDevice)) ||
      !okay(cudaMemcpy(state->angular_weight,
                       function->angular_weight.data(),
                       function->angular_weight.size() * sizeof(double),
                       cudaMemcpyHostToDevice))) {
    cudaFree(state->coefficients);
    cudaFree(state->expected_return);
    cudaFree(state->surface_ring_area);
    cudaFree(state->angular_weight);
    delete state;
    return 3;
  }
  function->cuda_state = state;
  return 0;
}

int apply_lxe_function_cuda(
    LXeFunctionInstance* function, std::uint64_t batch,
    const double* device_input, double* device_retained,
    double* device_egress, double* device_losses, void* stream_pointer) {
  if (!function || !function->cuda_state || !device_input ||
      !device_retained || !device_egress || !device_losses)
    return 1;
  auto& state = *static_cast<CudaState*>(function->cuda_state);
  if (!okay(cudaSetDevice(state.device)) ||
      !ensure_capacity(*function, state, batch))
    return 2;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  const auto states =
      function->nr * function->np * function->nm * function->nd;
  const auto phase_mode_count =
      batch * function->nr * function->nm * function->nd *
      function->orders;
  const auto modal_count =
      batch * function->orders * function->surface_radial;
  const auto surface_count =
      function->surface_radial * function->surface_phi;
  const auto egress_count = surface_count * function->angular;
  if (!okay(cudaMemsetAsync(device_retained, 0,
                            batch * states * sizeof(double), stream)) ||
      !okay(cudaMemsetAsync(state.expected_total, 0,
                            batch * sizeof(double), stream)))
    return 3;
  constexpr int threads = 256;
  phase_modes_kernel<<<
      static_cast<unsigned int>((phase_mode_count + threads - 1) / threads),
      threads, 0, stream>>>(
      device_input, batch, function->nr, function->np, function->nm,
      function->nd, function->orders, state.phase_modes);
  modal_surface_kernel<<<
      static_cast<unsigned int>((modal_count + threads - 1) / threads),
      threads, 0, stream>>>(
      state.phase_modes, state.coefficients, batch, function->nr,
      function->nm, function->nd, function->orders,
      function->surface_radial, state.modal_surface);
  input_totals_kernel<<<static_cast<unsigned int>(batch), threads, 0,
                        stream>>>(
      device_input, state.expected_return, batch, function->nr,
      function->np, function->nm, function->nd, state.input_total,
      state.expected_total);
  surface_weights_kernel<<<
      static_cast<unsigned int>(
          (batch * surface_count + threads - 1) / threads),
      threads, 0, stream>>>(
      state.modal_surface, state.surface_ring_area, batch, function->orders,
      function->surface_radial, function->surface_phi,
      state.surface_weight);
  egress_kernel<<<
      static_cast<unsigned int>(
          (batch * egress_count + threads - 1) / threads),
      threads, 0, stream>>>(
      state.surface_weight, state.angular_weight, batch, surface_count,
      function->angular,
      device_egress);
  losses_kernel<<<static_cast<unsigned int>((batch + threads - 1) / threads),
                  threads, 0, stream>>>(
      state.input_total, state.expected_total, batch, device_losses);
  return okay(cudaGetLastError()) ? 0 : 4;
}

int apply_lxe_function_adjoint_cuda(
    LXeFunctionInstance* function, std::uint64_t batch,
    const double* device_retained_adjoint,
    const double* device_egress_adjoint,
    const double* device_losses_adjoint, double* device_input_adjoint,
    void* stream_pointer) {
  if (!function || !function->cuda_state || !device_retained_adjoint ||
      !device_egress_adjoint || !device_losses_adjoint ||
      !device_input_adjoint)
    return 1;
  auto& state = *static_cast<CudaState*>(function->cuda_state);
  if (!okay(cudaSetDevice(state.device)) ||
      !ensure_capacity(*function, state, batch))
    return 2;
  (void)device_retained_adjoint;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  const auto states =
      function->nr * function->np * function->nm * function->nd;
  const auto phase_mode_count =
      batch * function->nr * function->nm * function->nd *
      function->orders;
  const auto modal_count =
      batch * function->orders * function->surface_radial;
  const auto surface_count =
      function->surface_radial * function->surface_phi;
  constexpr int threads = 256;
  egress_adjoint_surface_kernel<<<
      static_cast<unsigned int>(
          (batch * surface_count + threads - 1) / threads),
      threads, 0, stream>>>(
      device_egress_adjoint, state.angular_weight,
      state.surface_ring_area, batch, function->surface_radial,
      function->surface_phi, function->angular, state.surface_weight);
  surface_modes_adjoint_kernel<<<
      static_cast<unsigned int>((modal_count + threads - 1) / threads),
      threads, 0, stream>>>(
      state.surface_weight, batch, function->orders,
      function->surface_radial, function->surface_phi,
      state.modal_surface);
  phase_modes_adjoint_kernel<<<
      static_cast<unsigned int>((phase_mode_count + threads - 1) / threads),
      threads, 0, stream>>>(
      state.modal_surface, state.coefficients, batch, function->nr,
      function->nm, function->nd, function->orders,
      function->surface_radial, state.phase_modes);
  input_adjoint_kernel<<<
      static_cast<unsigned int>((batch * states + threads - 1) / threads),
      threads, 0, stream>>>(
      state.phase_modes, state.expected_return, device_losses_adjoint, batch,
      function->nr, function->np, function->nm, function->nd,
      function->orders, device_input_adjoint);
  return okay(cudaGetLastError()) ? 0 : 4;
}

void destroy_lxe_function_cuda(LXeFunctionInstance* function) {
  if (!function || !function->cuda_state) return;
  auto* state = static_cast<CudaState*>(function->cuda_state);
  cudaSetDevice(state->device);
  release_scratch(*state);
  cudaFree(state->coefficients);
  cudaFree(state->expected_return);
  cudaFree(state->surface_ring_area);
  cudaFree(state->angular_weight);
  delete state;
  function->cuda_state = nullptr;
}
