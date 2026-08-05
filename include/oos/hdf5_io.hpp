#pragma once

#include <filesystem>

#include "oos/effective_response.hpp"
#include "oos/scene.hpp"
#include "oos/solver.hpp"

namespace oos {

MeshData load_geometry_hdf5(const std::filesystem::path& path);
std::vector<std::uint32_t> load_surface_basis_id_hdf5(
    const std::filesystem::path& path, std::uint64_t triangle_count);
void save_geometry_hdf5(const std::filesystem::path& path,
                        const MeshData& mesh);
OperatorSet load_operators_hdf5(const std::filesystem::path& path);
void save_operators_hdf5(const std::filesystem::path& path,
                         const OperatorSet& operators);
SourceBatch load_source_batch_hdf5(const std::filesystem::path& path,
                                   std::uint64_t states,
                                   std::uint64_t channels,
                                   std::uint64_t losses);
void save_response_hdf5(const std::filesystem::path& path,
                        const SolveResult& result,
                        const OperatorSet& operators,
                        std::uint64_t source_count);

struct HitBatch {
  std::uint64_t count{};
  std::uint64_t channels{};
  std::vector<std::uint64_t> counts;
  // Optional emitted-photon count per event.  When present, regression uses
  // the absolute multinomial likelihood including the no-top-hit category;
  // otherwise it retains the conditional top-pattern likelihood.
  std::vector<std::uint64_t> emitted;
  std::vector<std::int32_t> channel_ids;
  std::vector<double> truth_xy_mm;
};

HitBatch load_hit_batch_hdf5(const std::filesystem::path& path);

}  // namespace oos
