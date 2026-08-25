#pragma once

#include <memory>

#include "oos/effective_response.hpp"
#include "oos/solver.hpp"

namespace oos {
struct ResponseGrid;
struct HitBatch;

SolveResult solve_cuda(const OperatorSet& operators, const SourceBatch& sources,
                       SolveControl control = {});
EffectiveResponse build_effective_response_cuda(
    const OperatorSet& operators, std::uint32_t cycles = 7,
    std::uint64_t batch_size = 64);

class CudaEffectiveResponseRuntime {
 public:
  explicit CudaEffectiveResponseRuntime(const EffectiveResponse& response);
  ~CudaEffectiveResponseRuntime();
  CudaEffectiveResponseRuntime(const CudaEffectiveResponseRuntime&) = delete;
  CudaEffectiveResponseRuntime& operator=(
      const CudaEffectiveResponseRuntime&) = delete;
  SolveResult apply(const SourceBatch& sources);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

SolveResult apply_effective_response_cuda(const EffectiveResponse& response,
                                          const SourceBatch& sources);
std::vector<double> score_response_grid_cuda(const ResponseGrid& grid,
                                             const HitBatch& hits);
}
