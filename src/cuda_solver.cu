#include "oos/cuda_solver.hpp"
#include "oos/function_operator.hpp"
#include "oos/regression.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <numeric>
#include <memory>
#include <vector>

namespace oos {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_sparse(cusparseStatus_t status, const char* operation) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

void check_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count = 0) : count_(count) {
    if (count_) check_cuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (data_) cudaFree(data_);
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
    return *this;
  }
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_{};
  std::size_t count_{};
};

struct DeviceCsr {
  explicit DeviceCsr(const CsrMatrix& host)
      : rows(host.rows),
        cols(host.cols),
        nonzeros(host.data.size()),
        indptr(host.indptr.size()),
        indices(host.indices.size()),
        values(host.data.size()) {
    std::vector<std::uint64_t> indices_64(host.indices.begin(),
                                          host.indices.end());
    check_cuda(cudaMemcpy(indptr.data(), host.indptr.data(),
                          host.indptr.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "copy CSR indptr");
    check_cuda(cudaMemcpy(indices.data(), indices_64.data(),
                          indices_64.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "copy CSR indices");
    check_cuda(cudaMemcpy(values.data(), host.data.data(),
                          host.data.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy CSR values");
    check_sparse(
        cusparseCreateCsr(
            &descriptor, static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
            static_cast<std::int64_t>(nonzeros), indptr.data(), indices.data(),
            values.data(), CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr");
  }
  ~DeviceCsr() {
    if (input_descriptor) cusparseDestroyDnMat(input_descriptor);
    if (output_descriptor) cusparseDestroyDnMat(output_descriptor);
    if (adjoint_input_descriptor)
      cusparseDestroyDnMat(adjoint_input_descriptor);
    if (adjoint_output_descriptor)
      cusparseDestroyDnMat(adjoint_output_descriptor);
    if (descriptor) cusparseDestroySpMat(descriptor);
  }
  DeviceCsr(const DeviceCsr&) = delete;
  DeviceCsr& operator=(const DeviceCsr&) = delete;

  std::uint64_t rows{};
  std::uint64_t cols{};
  std::uint64_t nonzeros{};
  DeviceBuffer<std::uint64_t> indptr;
  DeviceBuffer<std::uint64_t> indices;
  DeviceBuffer<double> values;
  DeviceBuffer<std::byte> spmm_workspace;
  DeviceBuffer<std::byte> adjoint_spmm_workspace;
  cusparseSpMatDescr_t descriptor{};
  cusparseDnMatDescr_t input_descriptor{};
  cusparseDnMatDescr_t output_descriptor{};
  std::uint64_t dense_batch{};
  bool spmm_prepared{};
  cusparseDnMatDescr_t adjoint_input_descriptor{};
  cusparseDnMatDescr_t adjoint_output_descriptor{};
  std::uint64_t adjoint_dense_batch{};
  bool adjoint_spmm_prepared{};
};

void right_multiply(cusparseHandle_t handle, DeviceCsr& matrix,
                    const double* input, std::uint64_t batch, double* output) {
  // A host row-major [batch, rows] is the same memory layout as a
  // column-major [rows, batch].  Compute A^T B so the output column-major
  // [cols, batch] is again host-compatible row-major [batch, cols].
  if (matrix.dense_batch != batch) {
    if (matrix.input_descriptor)
      cusparseDestroyDnMat(matrix.input_descriptor);
    if (matrix.output_descriptor)
      cusparseDestroyDnMat(matrix.output_descriptor);
    matrix.input_descriptor = nullptr;
    matrix.output_descriptor = nullptr;
    matrix.spmm_prepared = false;
    check_sparse(
        cusparseCreateDnMat(
            &matrix.input_descriptor,
            static_cast<std::int64_t>(matrix.rows),
            static_cast<std::int64_t>(batch),
            static_cast<std::int64_t>(matrix.rows),
            const_cast<double*>(input), CUDA_R_64F, CUSPARSE_ORDER_COL),
        "cusparseCreateDnMat(input)");
    check_sparse(cusparseCreateDnMat(
                     &matrix.output_descriptor,
                     static_cast<std::int64_t>(matrix.cols),
                     static_cast<std::int64_t>(batch),
                     static_cast<std::int64_t>(matrix.cols), output,
                     CUDA_R_64F, CUSPARSE_ORDER_COL),
                 "cusparseCreateDnMat(output)");
    matrix.dense_batch = batch;
  } else {
    check_sparse(cusparseDnMatSetValues(matrix.input_descriptor,
                                        const_cast<double*>(input)),
                 "cusparseDnMatSetValues(input)");
    check_sparse(cusparseDnMatSetValues(matrix.output_descriptor, output),
                 "cusparseDnMatSetValues(output)");
  }
  const double alpha = 1.0;
  const double beta = 0.0;
  if (!matrix.spmm_prepared) {
    std::size_t workspace_size{};
    check_sparse(
        cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix.descriptor,
            matrix.input_descriptor, &beta, matrix.output_descriptor,
            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size),
        "cusparseSpMM_bufferSize");
    if (matrix.spmm_workspace.size() < workspace_size)
      matrix.spmm_workspace = DeviceBuffer<std::byte>(workspace_size);
    matrix.spmm_prepared = true;
  }
  check_sparse(cusparseSpMM(
                   handle, CUSPARSE_OPERATION_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                   matrix.descriptor, matrix.input_descriptor, &beta,
                   matrix.output_descriptor, CUDA_R_64F,
                   CUSPARSE_SPMM_ALG_DEFAULT,
                   matrix.spmm_workspace.data()),
               "cusparseSpMM");
}

void adjoint_multiply(cusparseHandle_t handle, DeviceCsr& matrix,
                      const double* input, std::uint64_t batch,
                      double* output) {
  if (matrix.adjoint_dense_batch != batch) {
    if (matrix.adjoint_input_descriptor)
      cusparseDestroyDnMat(matrix.adjoint_input_descriptor);
    if (matrix.adjoint_output_descriptor)
      cusparseDestroyDnMat(matrix.adjoint_output_descriptor);
    matrix.adjoint_input_descriptor = nullptr;
    matrix.adjoint_output_descriptor = nullptr;
    matrix.adjoint_spmm_prepared = false;
    check_sparse(
        cusparseCreateDnMat(
            &matrix.adjoint_input_descriptor,
            static_cast<std::int64_t>(matrix.cols),
            static_cast<std::int64_t>(batch),
            static_cast<std::int64_t>(matrix.cols),
            const_cast<double*>(input), CUDA_R_64F, CUSPARSE_ORDER_COL),
        "cusparseCreateDnMat(adjoint input)");
    check_sparse(cusparseCreateDnMat(
                     &matrix.adjoint_output_descriptor,
                     static_cast<std::int64_t>(matrix.rows),
                     static_cast<std::int64_t>(batch),
                     static_cast<std::int64_t>(matrix.rows), output,
                     CUDA_R_64F, CUSPARSE_ORDER_COL),
                 "cusparseCreateDnMat(adjoint output)");
    matrix.adjoint_dense_batch = batch;
  } else {
    check_sparse(cusparseDnMatSetValues(
                     matrix.adjoint_input_descriptor,
                     const_cast<double*>(input)),
                 "cusparseDnMatSetValues(adjoint input)");
    check_sparse(cusparseDnMatSetValues(matrix.adjoint_output_descriptor,
                                        output),
                 "cusparseDnMatSetValues(adjoint output)");
  }
  const double alpha = 1.0;
  const double beta = 0.0;
  if (!matrix.adjoint_spmm_prepared) {
    std::size_t workspace_size{};
    check_sparse(
        cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix.descriptor,
            matrix.adjoint_input_descriptor, &beta,
            matrix.adjoint_output_descriptor, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size),
        "cusparseSpMM_bufferSize(adjoint)");
    if (matrix.adjoint_spmm_workspace.size() < workspace_size)
      matrix.adjoint_spmm_workspace = DeviceBuffer<std::byte>(workspace_size);
    matrix.adjoint_spmm_prepared = true;
  }
  check_sparse(
      cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                   matrix.descriptor, matrix.adjoint_input_descriptor, &beta,
                   matrix.adjoint_output_descriptor, CUDA_R_64F,
                   CUSPARSE_SPMM_ALG_DEFAULT,
                   matrix.adjoint_spmm_workspace.data()),
      "cusparseSpMM(adjoint)");
}

__global__ void row_sums(const double* values, std::uint64_t rows,
                         std::uint64_t columns, double* result) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  double local = 0.0;
  for (std::uint64_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    local += values[row * columns + column];
  }
  __shared__ double partial[256];
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) result[row] = partial[0];
}

__global__ void gather_state_slice(
    const double* global, std::uint64_t batch,
    std::uint64_t global_states, std::uint64_t offset,
    std::uint64_t count, double* local) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = batch * count;
  if (linear >= total) return;
  const auto row = linear / count;
  const auto state = linear % count;
  local[linear] = global[row * global_states + offset + state];
}

__global__ void add_state_slice(
    const double* local, std::uint64_t batch, std::uint64_t count,
    std::uint64_t offset, std::uint64_t global_states, double* global) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = batch * count;
  if (linear >= total) return;
  const auto row = linear / count;
  const auto state = linear % count;
  global[row * global_states + offset + state] += local[linear];
}

__global__ void add_intrinsic_losses(
    const double* local, std::uint64_t batch, std::uint64_t local_count,
    const std::uint32_t* columns, std::uint64_t global_count,
    double* global) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = batch * local_count;
  if (linear >= total) return;
  const auto row = linear / local_count;
  const auto loss = linear % local_count;
  global[row * global_count + columns[loss]] += local[linear];
}

__global__ void gather_intrinsic_losses(
    const double* global, std::uint64_t batch, std::uint64_t local_count,
    const std::uint32_t* columns, std::uint64_t global_count,
    double* local) {
  const auto linear =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = batch * local_count;
  if (linear >= total) return;
  const auto row = linear / local_count;
  const auto loss = linear % local_count;
  local[linear] = global[row * global_count + columns[loss]];
}

struct DeviceFunctionBlock {
  DeviceFunctionBlock(const FunctionBlock& source, double energy_eV,
                      double tolerance, std::uint64_t batch,
                      std::uint64_t global_states,
                      std::uint64_t global_channels,
                      std::uint64_t global_losses)
      : block(source),
        function(std::make_unique<FunctionOperator>(
            source.library_path, source.config_json, energy_eV)),
        egress_to_transition(source.egress_to_transition),
        egress_to_detection(source.egress_to_detection),
        egress_to_losses(source.egress_to_losses),
        input(batch * source.state_count),
        retained(batch * source.state_count),
        egress(batch * source.egress_count),
        adjoint_egress_scratch(batch * source.egress_count),
        intrinsic_losses(batch * source.intrinsic_loss_count),
        coupled_state(batch * global_states),
        coupled_detection(batch * global_channels),
        coupled_losses(batch * global_losses),
        intrinsic_loss_columns(source.intrinsic_loss_columns.size()) {
    const auto& descriptor = function->descriptor();
    if (descriptor.input_state_count != block.state_count ||
        descriptor.retained_state_count != block.state_count ||
        descriptor.egress_count != block.egress_count ||
        descriptor.loss_count != block.intrinsic_loss_count ||
        std::abs(descriptor.contraction_bound - block.contraction_bound) >
            tolerance)
      throw std::runtime_error(
          "functional block CUDA descriptor does not match cache");
    check_cuda(
        cudaMemcpy(intrinsic_loss_columns.data(),
                   block.intrinsic_loss_columns.data(),
                   block.intrinsic_loss_columns.size() *
                       sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
        "copy functional intrinsic loss columns");
    function->prepare_cuda(0);
  }

  const FunctionBlock& block;
  std::unique_ptr<FunctionOperator> function;
  DeviceCsr egress_to_transition;
  DeviceCsr egress_to_detection;
  DeviceCsr egress_to_losses;
  DeviceBuffer<double> input;
  DeviceBuffer<double> retained;
  DeviceBuffer<double> egress;
  DeviceBuffer<double> adjoint_egress_scratch;
  DeviceBuffer<double> intrinsic_losses;
  DeviceBuffer<double> coupled_state;
  DeviceBuffer<double> coupled_detection;
  DeviceBuffer<double> coupled_losses;
  DeviceBuffer<std::uint32_t> intrinsic_loss_columns;
};

}  // namespace

SolveResult solve_cuda(const OperatorSet& operators,
                       const SourceBatch& sources,
                       SolveControl control) {
  const auto started = std::chrono::steady_clock::now();
  operators.validate();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    throw std::runtime_error("CUDA reports no available device");
  }
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0),
             "cudaGetDeviceProperties");
  const auto states = operators.transition.rows;
  const auto channels = operators.detection.cols;
  const auto losses = operators.losses.cols;
  if (sources.initial_states.size() != sources.count * states ||
      sources.direct_detection.size() != sources.count * channels ||
      sources.direct_losses.size() != sources.count * losses) {
    throw std::runtime_error("source batch shape mismatch");
  }
  cusparseHandle_t sparse{};
  cublasHandle_t blas{};
  check_sparse(cusparseCreate(&sparse), "cusparseCreate");
  check_blas(cublasCreate(&blas), "cublasCreate");
  try {
    DeviceCsr transition(operators.transition);
    DeviceCsr detection(operators.detection);
    DeviceCsr loss_operator(operators.losses);
    DeviceBuffer<double> current(sources.count * states);
    DeviceBuffer<double> next(sources.count * states);
    DeviceBuffer<double> detected(sources.count * channels);
    DeviceBuffer<double> lost(sources.count * losses);
    DeviceBuffer<double> efficiency(sources.count * channels);
    DeviceBuffer<double> loss_totals(sources.count * losses);
    DeviceBuffer<double> unresolved(sources.count);
    std::vector<std::unique_ptr<DeviceFunctionBlock>> function_blocks;
    function_blocks.reserve(operators.function_blocks.size());
    for (const auto& block : operators.function_blocks)
      function_blocks.push_back(std::make_unique<DeviceFunctionBlock>(
          block, operators.energy_eV, operators.tolerance, sources.count,
          states, channels, losses));
    check_cuda(cudaMemcpy(current.data(), sources.initial_states.data(),
                          current.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy initial states");
    check_cuda(cudaMemcpy(efficiency.data(),
                          sources.direct_detection.data(),
                          efficiency.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy direct detection");
    check_cuda(cudaMemcpy(loss_totals.data(), sources.direct_losses.data(),
                          loss_totals.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy direct losses");

    SolveResult result;
    result.backend = "cuda";
    result.hardware = std::string(properties.name) + "; cuda=" +
                      std::to_string(CUDART_VERSION);
    result.unresolved.resize(sources.count);
    result.input_weight.resize(sources.count);
    result.source_integration_l1_error_estimate =
        sources.source_integration_l1_error_estimate;
    if (result.source_integration_l1_error_estimate.empty())
      result.source_integration_l1_error_estimate.assign(sources.count, 0.0);
    if (result.source_integration_l1_error_estimate.size() != sources.count)
      throw std::runtime_error(
          "source integration error-estimate shape mismatch");
    for (std::uint64_t source = 0; source < sources.count; ++source) {
      result.input_weight[source] =
          std::accumulate(
              sources.initial_states.begin() + source * states,
              sources.initial_states.begin() + (source + 1) * states, 0.0) +
          std::accumulate(
              sources.direct_detection.begin() + source * channels,
              sources.direct_detection.begin() + (source + 1) * channels,
              0.0) +
          std::accumulate(
              sources.direct_losses.begin() + source * losses,
              sources.direct_losses.begin() + (source + 1) * losses, 0.0);
    }
    const double one = 1.0;
    bool converged = false;
    const auto maximum_iterations =
        control.maximum_iterations == 0 ? operators.maximum_iterations
                                        : control.maximum_iterations;
    if (maximum_iterations == 0)
      throw std::runtime_error("solve iteration limit must be positive");
    for (std::uint32_t iteration = 0; iteration < maximum_iterations;
         ++iteration) {
      right_multiply(sparse, detection, current.data(), sources.count,
                     detected.data());
      right_multiply(sparse, loss_operator, current.data(), sources.count,
                     lost.data());
      check_blas(cublasDaxpy(
                     blas, static_cast<int>(efficiency.size()), &one,
                     detected.data(), 1, efficiency.data(), 1),
                 "accumulate detection");
      check_blas(cublasDaxpy(blas, static_cast<int>(loss_totals.size()), &one,
                             lost.data(), 1, loss_totals.data(), 1),
                 "accumulate losses");
      right_multiply(sparse, transition, current.data(), sources.count,
                     next.data());
      constexpr int threads = 256;
      for (auto& runtime : function_blocks) {
        const auto& block = runtime->block;
        const auto state_values = sources.count * block.state_count;
        gather_state_slice<<<
            static_cast<unsigned int>(
                (state_values + threads - 1) / threads),
            threads>>>(current.data(), sources.count, states,
                       block.state_offset, block.state_count,
                       runtime->input.data());
        check_cuda(cudaGetLastError(), "gather functional state slice");
        runtime->function->apply_cuda(
            sources.count, runtime->input.data(), runtime->retained.data(),
            runtime->egress.data(), runtime->intrinsic_losses.data());
        add_state_slice<<<
            static_cast<unsigned int>(
                (state_values + threads - 1) / threads),
            threads>>>(runtime->retained.data(), sources.count,
                       block.state_count, block.state_offset, states,
                       next.data());
        check_cuda(cudaGetLastError(), "add functional retained states");

        right_multiply(sparse, runtime->egress_to_transition,
                       runtime->egress.data(), sources.count,
                       runtime->coupled_state.data());
        right_multiply(sparse, runtime->egress_to_detection,
                       runtime->egress.data(), sources.count,
                       runtime->coupled_detection.data());
        right_multiply(sparse, runtime->egress_to_losses,
                       runtime->egress.data(), sources.count,
                       runtime->coupled_losses.data());
        check_blas(cublasDaxpy(
                       blas, static_cast<int>(runtime->coupled_state.size()),
                       &one, runtime->coupled_state.data(), 1, next.data(), 1),
                   "accumulate functional states");
        check_blas(cublasDaxpy(
                       blas,
                       static_cast<int>(runtime->coupled_detection.size()),
                       &one, runtime->coupled_detection.data(), 1,
                       efficiency.data(), 1),
                   "accumulate functional detection");
        check_blas(cublasDaxpy(
                       blas, static_cast<int>(runtime->coupled_losses.size()),
                       &one, runtime->coupled_losses.data(), 1,
                       loss_totals.data(), 1),
                   "accumulate functional coupled losses");
        const auto intrinsic_values =
            sources.count * block.intrinsic_loss_count;
        add_intrinsic_losses<<<
            static_cast<unsigned int>(
                (intrinsic_values + threads - 1) / threads),
            threads>>>(
            runtime->intrinsic_losses.data(), sources.count,
            block.intrinsic_loss_count,
            runtime->intrinsic_loss_columns.data(), losses,
            loss_totals.data());
        check_cuda(cudaGetLastError(),
                   "accumulate functional intrinsic losses");
      }
      row_sums<<<static_cast<unsigned int>(sources.count), 256>>>(
          next.data(), sources.count, states, unresolved.data());
      check_cuda(cudaGetLastError(), "row_sums");
      check_cuda(cudaMemcpy(result.unresolved.data(), unresolved.data(),
                            sources.count * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "copy unresolved");
      result.iterations = iteration + 1;
      std::swap(current, next);
      const double maximum =
          *std::max_element(result.unresolved.begin(),
                            result.unresolved.end());
      if (maximum <= operators.tolerance) {
        converged = true;
        break;
      }
    }
    if (!converged && control.require_convergence)
      throw std::runtime_error("CUDA Neumann solve did not converge");
    result.efficiency.resize(efficiency.size());
    result.losses.resize(loss_totals.size());
    check_cuda(cudaMemcpy(result.efficiency.data(), efficiency.data(),
                          efficiency.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy efficiency");
    check_cuda(cudaMemcpy(result.losses.data(), loss_totals.data(),
                          loss_totals.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy losses");
    populate_float32_efficiency_loss_upper_bound(operators, sources.count,
                                                  result);
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    result.peak_device_bytes = total_bytes - free_bytes;
    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    cublasDestroy(blas);
    cusparseDestroy(sparse);
    return result;
  } catch (...) {
    cublasDestroy(blas);
    cusparseDestroy(sparse);
    throw;
  }
}

EffectiveResponse build_effective_response_cuda(
    const OperatorSet& operators, std::uint32_t cycles,
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
  response.construction_method = "adjoint_linear";
  response.state_to_detection.assign(response.states * response.channels,
                                     0.0);
  response.state_to_losses.assign(response.states * response.losses, 0.0);
  response.state_unresolved.assign(response.states, 0.0);
  response.channel_ids = operators.channel_ids;
  response.loss_names = operators.loss_names;
  response.operator_cache_key_sha256 = operators.cache_key_sha256;
  response.code_commit = operators.code_commit;
  cusparseHandle_t sparse{};
  cublasHandle_t blas{};
  check_sparse(cusparseCreate(&sparse), "cusparseCreate");
  check_blas(cublasCreate(&blas), "cublasCreate");
  try {
    DeviceCsr transition(operators.transition);
    DeviceCsr detection(operators.detection);
    DeviceCsr loss_operator(operators.losses);
    DeviceBuffer<double> next_state(batch_size * response.states);
    DeviceBuffer<double> current_state(batch_size * response.states);
    DeviceBuffer<double> state_scratch(batch_size * response.states);
    DeviceBuffer<double> detection_seed(batch_size * response.channels);
    DeviceBuffer<double> loss_seed(batch_size * response.losses);
    std::vector<std::unique_ptr<DeviceFunctionBlock>> function_blocks;
    function_blocks.reserve(operators.function_blocks.size());
    for (const auto& block : operators.function_blocks) {
      auto runtime = std::make_unique<DeviceFunctionBlock>(
          block, operators.energy_eV, operators.tolerance, batch_size,
          response.states, response.channels,
          response.losses);
      function_blocks.push_back(std::move(runtime));
    }
    constexpr int threads = 256;
    const double one = 1.0;
    const auto apply_cycle = [&](std::uint64_t count) {
      adjoint_multiply(sparse, transition, next_state.data(), count,
                       current_state.data());
      adjoint_multiply(sparse, detection, detection_seed.data(), count,
                       state_scratch.data());
      check_blas(cublasDaxpy(
                     blas, static_cast<int>(count * response.states), &one,
                     state_scratch.data(), 1, current_state.data(), 1),
                 "accumulate adjoint direct detection");
      adjoint_multiply(sparse, loss_operator, loss_seed.data(), count,
                       state_scratch.data());
      check_blas(cublasDaxpy(
                     blas, static_cast<int>(count * response.states), &one,
                     state_scratch.data(), 1, current_state.data(), 1),
                 "accumulate adjoint direct losses");
      for (auto& runtime : function_blocks) {
        const auto& block = runtime->block;
        const auto state_values = count * block.state_count;
        gather_state_slice<<<
            static_cast<unsigned int>(
                (state_values + threads - 1) / threads),
            threads>>>(next_state.data(), count, response.states,
                       block.state_offset, block.state_count,
                       runtime->retained.data());
        check_cuda(cudaGetLastError(),
                   "gather adjoint functional retained values");
        adjoint_multiply(sparse, runtime->egress_to_transition,
                         next_state.data(), count, runtime->egress.data());
        adjoint_multiply(sparse, runtime->egress_to_detection,
                         detection_seed.data(), count,
                         runtime->adjoint_egress_scratch.data());
        check_blas(cublasDaxpy(
                       blas, static_cast<int>(count * block.egress_count),
                       &one, runtime->adjoint_egress_scratch.data(), 1,
                       runtime->egress.data(), 1),
                   "accumulate adjoint functional detection egress");
        adjoint_multiply(sparse, runtime->egress_to_losses,
                         loss_seed.data(), count,
                         runtime->adjoint_egress_scratch.data());
        check_blas(cublasDaxpy(
                       blas, static_cast<int>(count * block.egress_count),
                       &one, runtime->adjoint_egress_scratch.data(), 1,
                       runtime->egress.data(), 1),
                   "accumulate adjoint functional loss egress");
        const auto intrinsic_values =
            count * block.intrinsic_loss_count;
        if (intrinsic_values != 0) {
          gather_intrinsic_losses<<<
              static_cast<unsigned int>(
                  (intrinsic_values + threads - 1) / threads),
              threads>>>(loss_seed.data(), count,
                         block.intrinsic_loss_count,
                         runtime->intrinsic_loss_columns.data(),
                         response.losses, runtime->intrinsic_losses.data());
          check_cuda(cudaGetLastError(),
                     "gather adjoint functional intrinsic losses");
        }
        runtime->function->apply_adjoint_cuda(
            count, runtime->retained.data(), runtime->egress.data(),
            runtime->intrinsic_losses.data(), runtime->input.data());
        add_state_slice<<<
            static_cast<unsigned int>(
                (state_values + threads - 1) / threads),
            threads>>>(runtime->input.data(), count, block.state_count,
                       block.state_offset, response.states,
                       current_state.data());
        check_cuda(cudaGetLastError(),
                   "accumulate adjoint functional input states");
      }
      std::swap(next_state, current_state);
    };

    std::vector<double> host_detection_seed(batch_size * response.channels,
                                            0.0);
    std::vector<double> host_loss_seed(batch_size * response.losses, 0.0);
    std::vector<double> host_state(batch_size * response.states, 0.0);
    const auto build_terminal =
        [&](std::uint64_t terminal_count, bool is_detection,
            std::vector<double>& output) {
          for (std::uint64_t start = 0; start < terminal_count;
               start += batch_size) {
            const auto count = std::min(batch_size, terminal_count - start);
            std::fill(host_detection_seed.begin(),
                      host_detection_seed.end(), 0.0);
            std::fill(host_loss_seed.begin(), host_loss_seed.end(), 0.0);
            auto& host_seed =
                is_detection ? host_detection_seed : host_loss_seed;
            const auto columns =
                is_detection ? response.channels : response.losses;
            for (std::uint64_t row = 0; row < count; ++row)
              host_seed[row * columns + start + row] = 1.0;
            check_cuda(cudaMemcpy(detection_seed.data(),
                                  host_detection_seed.data(),
                                  count * response.channels * sizeof(double),
                                  cudaMemcpyHostToDevice),
                       "copy adjoint detection seeds");
            check_cuda(cudaMemcpy(loss_seed.data(), host_loss_seed.data(),
                                  count * response.losses * sizeof(double),
                                  cudaMemcpyHostToDevice),
                       "copy adjoint loss seeds");
            check_cuda(cudaMemset(next_state.data(), 0,
                                  count * response.states * sizeof(double)),
                       "clear adjoint terminal state");
            for (std::uint32_t cycle = 0; cycle < cycles; ++cycle)
              apply_cycle(count);
            check_cuda(cudaMemcpy(host_state.data(), next_state.data(),
                                  count * response.states * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "copy adjoint terminal response");
            for (std::uint64_t state = 0; state < response.states; ++state)
              for (std::uint64_t row = 0; row < count; ++row)
                output[state * terminal_count + start + row] =
                    host_state[row * response.states + state];
          }
        };
    build_terminal(response.channels, true, response.state_to_detection);
    build_terminal(response.losses, false, response.state_to_losses);

    std::fill(host_state.begin(), host_state.end(), 0.0);
    std::fill_n(host_state.begin(), response.states, 1.0);
    check_cuda(cudaMemcpy(next_state.data(), host_state.data(),
                          response.states * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy adjoint unresolved terminal");
    check_cuda(cudaMemset(detection_seed.data(), 0,
                          response.channels * sizeof(double)),
               "clear adjoint unresolved detection seeds");
    check_cuda(cudaMemset(loss_seed.data(), 0,
                          response.losses * sizeof(double)),
               "clear adjoint unresolved loss seeds");
    for (std::uint32_t cycle = 0; cycle < cycles; ++cycle) apply_cycle(1);
    check_cuda(cudaMemcpy(response.state_unresolved.data(), next_state.data(),
                          response.states * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy adjoint unresolved response");
    cublasDestroy(blas);
    cusparseDestroy(sparse);
  } catch (...) {
    cublasDestroy(blas);
    cusparseDestroy(sparse);
    throw;
  }
  response.validate();
  return response;
}

struct CudaEffectiveResponseRuntime::Impl {
  explicit Impl(const EffectiveResponse& value)
      : response(value),
        detection(value.state_to_detection.size()),
        loss_matrix(value.state_to_losses.size()),
        unresolved_vector(value.state_unresolved.size()) {
    response.validate();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
      throw std::runtime_error("CUDA reports no available device");
    check_cuda(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties");
    check_blas(cublasCreate(&blas), "cublasCreate");
    check_cuda(cudaMemcpy(detection.data(),
                          response.state_to_detection.data(),
                          detection.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy effective detection response");
    check_cuda(cudaMemcpy(loss_matrix.data(), response.state_to_losses.data(),
                          loss_matrix.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy effective loss response");
    check_cuda(cudaMemcpy(unresolved_vector.data(),
                          response.state_unresolved.data(),
                          unresolved_vector.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy effective unresolved response");
  }

  ~Impl() {
    if (blas) cublasDestroy(blas);
  }

  const EffectiveResponse& response;
  DeviceBuffer<double> detection;
  DeviceBuffer<double> loss_matrix;
  DeviceBuffer<double> unresolved_vector;
  cudaDeviceProp properties{};
  cublasHandle_t blas{};
};

CudaEffectiveResponseRuntime::CudaEffectiveResponseRuntime(
    const EffectiveResponse& response)
    : impl_(std::make_unique<Impl>(response)) {}

CudaEffectiveResponseRuntime::~CudaEffectiveResponseRuntime() = default;

SolveResult CudaEffectiveResponseRuntime::apply(const SourceBatch& sources) {
  const auto started = std::chrono::steady_clock::now();
  const auto& response = impl_->response;
  if (sources.initial_states.size() != sources.count * response.states ||
      sources.direct_detection.size() != sources.count * response.channels ||
      sources.direct_losses.size() != sources.count * response.losses)
    throw std::runtime_error("source batch shape mismatch");
  DeviceBuffer<double> source(sources.initial_states.size());
  DeviceBuffer<double> efficiency(sources.count * response.channels);
  DeviceBuffer<double> losses(sources.count * response.losses);
  DeviceBuffer<double> unresolved(sources.count);
    check_cuda(cudaMemcpy(source.data(), sources.initial_states.data(),
                          source.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy effective source states");
    check_cuda(cudaMemcpy(efficiency.data(),
                          sources.direct_detection.data(),
                          efficiency.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy direct efficiency");
    check_cuda(cudaMemcpy(losses.data(), sources.direct_losses.data(),
                          losses.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "copy direct losses");
    check_cuda(cudaMemset(unresolved.data(), 0,
                          unresolved.size() * sizeof(double)),
               "clear unresolved");
    const double one = 1.0;
    const double zero = 0.0;
    check_blas(
        cublasDgemm(
            impl_->blas, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(response.channels),
            static_cast<int>(sources.count),
            static_cast<int>(response.states), &one, impl_->detection.data(),
            static_cast<int>(response.channels), source.data(),
            static_cast<int>(response.states), &one, efficiency.data(),
            static_cast<int>(response.channels)),
        "apply effective detection response");
    if (response.losses != 0) {
      check_blas(
          cublasDgemm(
              impl_->blas, CUBLAS_OP_N, CUBLAS_OP_N,
              static_cast<int>(response.losses),
              static_cast<int>(sources.count),
              static_cast<int>(response.states), &one,
              impl_->loss_matrix.data(),
              static_cast<int>(response.losses), source.data(),
              static_cast<int>(response.states), &one, losses.data(),
              static_cast<int>(response.losses)),
          "apply effective loss response");
    }
    check_blas(
        cublasDgemv(impl_->blas, CUBLAS_OP_T,
                    static_cast<int>(response.states),
                    static_cast<int>(sources.count), &one, source.data(),
                    static_cast<int>(response.states),
                    impl_->unresolved_vector.data(), 1, &zero,
                    unresolved.data(), 1),
        "apply effective unresolved response");
    SolveResult result;
    result.backend = "cuda-precomputed-adjoint";
    result.hardware = std::string(impl_->properties.name) + "; cuda=" +
                      std::to_string(CUDART_VERSION);
    result.efficiency.resize(efficiency.size());
    result.losses.resize(losses.size());
    result.unresolved.resize(sources.count);
    check_cuda(cudaMemcpy(result.efficiency.data(), efficiency.data(),
                          efficiency.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy effective efficiency");
    check_cuda(cudaMemcpy(result.losses.data(), losses.data(),
                          losses.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy effective losses");
    check_cuda(cudaMemcpy(result.unresolved.data(), unresolved.data(),
                          unresolved.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy effective unresolved");
    result.input_weight.assign(sources.count, 0.0);
    for (std::uint64_t source_index = 0; source_index < sources.count;
         ++source_index) {
      result.input_weight[source_index] =
          std::accumulate(
              sources.initial_states.begin() +
                  source_index * response.states,
              sources.initial_states.begin() +
                  (source_index + 1) * response.states,
              0.0) +
          std::accumulate(
              sources.direct_detection.begin() +
                  source_index * response.channels,
              sources.direct_detection.begin() +
                  (source_index + 1) * response.channels,
              0.0) +
          std::accumulate(
              sources.direct_losses.begin() +
                  source_index * response.losses,
              sources.direct_losses.begin() +
                  (source_index + 1) * response.losses,
              0.0);
    }
    result.source_integration_l1_error_estimate =
        sources.source_integration_l1_error_estimate;
    if (result.source_integration_l1_error_estimate.empty())
      result.source_integration_l1_error_estimate.assign(sources.count, 0.0);
    result.iterations = response.cycles;
    OperatorSet audit;
    audit.loss_names = response.loss_names;
    audit.losses.cols = response.losses;
    populate_float32_efficiency_loss_upper_bound(audit, sources.count, result);
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    result.peak_device_bytes = total_bytes - free_bytes;
    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    return result;
}

SolveResult apply_effective_response_cuda(const EffectiveResponse& response,
                                          const SourceBatch& sources) {
  CudaEffectiveResponseRuntime runtime(response);
  return runtime.apply(sources);
}

std::vector<double> score_response_grid_cuda(const ResponseGrid& grid,
                                             const HitBatch& hits) {
  grid.validate();
  if (hits.channels != grid.channels ||
      hits.channel_ids != grid.channel_ids)
    throw std::runtime_error("hit channels do not match response grid");
  std::vector<double> log_probability(
      grid.conditional_log_probability.begin(),
      grid.conditional_log_probability.end());
  std::vector<double> counts(hits.counts.begin(), hits.counts.end());
  DeviceBuffer<double> device_log_probability(log_probability.size());
  DeviceBuffer<double> device_counts(counts.size());
  DeviceBuffer<double> device_result(hits.count * grid.points);
  check_cuda(cudaMemcpy(device_log_probability.data(), log_probability.data(),
                        log_probability.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "copy response-grid log probabilities");
  check_cuda(cudaMemcpy(device_counts.data(), counts.data(),
                        counts.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "copy channel counts");
  cublasHandle_t blas{};
  check_blas(cublasCreate(&blas), "cublasCreate");
  try {
    const double one = 1.0;
    const double zero = 0.0;
    // Row-major [points,channels] and [events,channels] appear as
    // column-major [channels,points] and [channels,events].
    check_blas(
        cublasDgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(grid.points), static_cast<int>(hits.count),
            static_cast<int>(grid.channels), &one,
            device_log_probability.data(), static_cast<int>(grid.channels),
            device_counts.data(), static_cast<int>(grid.channels), &zero,
            device_result.data(), static_cast<int>(grid.points)),
        "score full-plane likelihood");
    std::vector<double> result(hits.count * grid.points);
    check_cuda(cudaMemcpy(result.data(), device_result.data(),
                          result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "copy full-plane likelihood");
    cublasDestroy(blas);
    return result;
  } catch (...) {
    cublasDestroy(blas);
    throw;
  }
}

}  // namespace oos
