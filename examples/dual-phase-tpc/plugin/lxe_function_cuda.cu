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
  double* positive_total{};
};

bool okay(cudaError_t status) { return status == cudaSuccess; }

void release_scratch(CudaState& state) {
  cudaFree(state.phase_modes);
  cudaFree(state.modal_surface);
  cudaFree(state.surface_weight);
  cudaFree(state.input_total);
  cudaFree(state.expected_total);
  cudaFree(state.positive_total);
  state.phase_modes = nullptr;
  state.modal_surface = nullptr;
  state.surface_weight = nullptr;
  state.input_total = nullptr;
  state.expected_total = nullptr;
  state.positive_total = nullptr;
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
      !okay(cudaMalloc(&state.expected_total, batch * sizeof(double))) ||
      !okay(cudaMalloc(&state.positive_total, batch * sizeof(double)))) {
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
  const double raw =
      cuCreal(density) * ring_area[surface] /
      static_cast<double>(surface_phi);
  const double positive = fmax(0.0, raw);
  surface_weight[linear] = positive;
}

__global__ void positive_totals_kernel(
    const double* surface_weight, std::uint64_t batch,
    std::uint64_t surface_count, double* positive_total) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= batch) return;
  double local = 0.0;
  for (std::uint64_t surface = threadIdx.x; surface < surface_count;
       surface += blockDim.x)
    local += surface_weight[row * surface_count + surface];
  __shared__ double partial[256];
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) positive_total[row] = partial[0];
}

__global__ void normalize_egress_kernel(
    const double* surface_weight, const double* expected_total,
    const double* positive_total, const double* angular_weight,
    std::uint64_t batch, std::uint64_t surface_count,
    std::uint64_t angular, double* egress) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch * surface_count * angular;
  if (linear >= count) return;
  auto value = linear;
  const auto angle = value % angular;
  value /= angular;
  const auto surface = value % surface_count;
  const auto row = value / surface_count;
  const double normalization =
      positive_total[row] > 0.0
          ? expected_total[row] / positive_total[row]
          : 0.0;
  egress[linear] = surface_weight[row * surface_count + surface] *
                   normalization * angular_weight[angle];
}

__global__ void losses_kernel(const double* input_total,
                              const double* expected_total,
                              std::uint64_t batch, double* losses) {
  const auto row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < batch)
    losses[row] = fmax(0.0, input_total[row] - expected_total[row]);
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
    double* device_egress, double* device_losses, void* stream_pointer,
    oos_function_operator_audit_v1* host_audit) {
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
  positive_totals_kernel<<<static_cast<unsigned int>(batch), threads, 0,
                            stream>>>(
      state.surface_weight, batch, surface_count, state.positive_total);
  normalize_egress_kernel<<<
      static_cast<unsigned int>(
          (batch * egress_count + threads - 1) / threads),
      threads, 0, stream>>>(
      state.surface_weight, state.expected_total, state.positive_total,
      state.angular_weight, batch, surface_count, function->angular,
      device_egress);
  losses_kernel<<<static_cast<unsigned int>((batch + threads - 1) / threads),
                  threads, 0, stream>>>(
      state.input_total, state.expected_total, batch, device_losses);
  if (!okay(cudaGetLastError())) return 4;
  if (host_audit) {
    std::vector<double> input_total(batch);
    std::vector<double> expected_total(batch);
    std::vector<double> losses(batch);
    if (!okay(cudaMemcpyAsync(input_total.data(), state.input_total,
                              batch * sizeof(double),
                              cudaMemcpyDeviceToHost, stream)) ||
        !okay(cudaMemcpyAsync(expected_total.data(), state.expected_total,
                              batch * sizeof(double),
                              cudaMemcpyDeviceToHost, stream)) ||
        !okay(cudaMemcpyAsync(losses.data(), device_losses,
                              batch * sizeof(double),
                              cudaMemcpyDeviceToHost, stream)) ||
        !okay(cudaStreamSynchronize(stream)))
      return 5;
    for (std::uint64_t row = 0; row < batch; ++row)
      host_audit[row] = {
          input_total[row], 0.0, expected_total[row], losses[row],
          expected_total[row] + losses[row] - input_total[row]};
  }
  return 0;
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
