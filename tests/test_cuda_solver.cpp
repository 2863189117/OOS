#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "oos/cuda_solver.hpp"
#include "oos/effective_response.hpp"
#include "oos/regression.hpp"

namespace {

void group(hid_t file, const char* path) {
  const auto handle = H5Gcreate2(file, path, H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT);
  REQUIRE(handle >= 0);
  H5Gclose(handle);
}

template <typename T>
void dataset(hid_t file, const char* path, hid_t type,
             const std::vector<T>& values,
             const std::vector<hsize_t>& dimensions) {
  const auto space =
      H5Screate_simple(static_cast<int>(dimensions.size()),
                       dimensions.data(), nullptr);
  const auto handle =
      H5Dcreate2(file, path, type, space, H5P_DEFAULT, H5P_DEFAULT,
                 H5P_DEFAULT);
  REQUIRE(handle >= 0);
  REQUIRE(H5Dwrite(handle, type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   values.data()) >= 0);
  H5Dclose(handle);
  H5Sclose(space);
}

void write_factorized_block(const std::filesystem::path& path) {
  const auto file =
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);
  group(file, "/nonlocal");
  group(file, "/nonlocal/egress");
  group(file, "/function");
  group(file, "/metadata");
  dataset(file, "/nonlocal/egress/surface_element", H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{0}, {1});
  dataset(file, "/nonlocal/egress/barycentric", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, {1, 3});
  dataset(file, "/nonlocal/egress/side", H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{0}, {1});
  dataset(file, "/nonlocal/egress/direction_local", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0, 0.0, 1.0}, {1, 3});
  dataset(file, "/nonlocal/egress/stokes", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0, 0.0, 0.0, 0.0}, {1, 4});
  dataset(file, "/nonlocal/egress/reference_axis_local",
          H5T_NATIVE_DOUBLE, std::vector<double>{1.0, 0.0, 0.0},
          {1, 3});
  dataset(file, "/function/coefficients_real", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.4}, {1, 1, 1, 1, 1});
  dataset(file, "/function/coefficients_imag", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0}, {1, 1, 1, 1, 1});
  dataset(file, "/function/expected_return", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.4}, {1, 1, 1});
  dataset(file, "/function/audit_values", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.6}, {1, 1, 1, 1});
  dataset(file, "/function/surface_radius_mm", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0}, {1});
  dataset(file, "/function/surface_ring_area_mm2", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0}, {1});
  dataset(file, "/function/angular_weight", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0}, {1});
  const std::string losses = R"(["lxe_nonreturn"])";
  const std::string audits = R"(["lxe_nonreturn"])";
  const std::string grid =
      R"({"position_radial_bins":1,"position_phi_bins":1,"direction_mu_bins":1,"direction_phi_bins":1})";
  const std::string generator =
      R"({"schema":"oos.nonlocal.function.v1","state_count":1,"egress_count":1,"surface_phi_bins":1,"angular_count":1,"contraction_bound":0.4})";
  dataset(file, "/metadata/loss_names_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(losses.begin(), losses.end()),
          {losses.size()});
  dataset(file, "/metadata/audit_names_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(audits.begin(), audits.end()),
          {audits.size()});
  dataset(file, "/metadata/phase_grid_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(grid.begin(), grid.end()),
          {grid.size()});
  dataset(file, "/metadata/generator_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(generator.begin(), generator.end()),
          {generator.size()});
  H5Fclose(file);
}

}  // namespace

TEST_CASE("CUDA Neumann solver matches the CPU batch result") {
  oos::OperatorSet operators;
  operators.transition = {2, 2, {0, 1, 2}, {1, 0}, {0.2, 0.1}};
  operators.detection = {2, 2, {0, 1, 2}, {0, 1}, {0.5, 0.6}};
  operators.losses = {2, 1, {0, 1, 2}, {0, 0}, {0.3, 0.3}};
  operators.channel_ids = {10, 20};
  operators.loss_names = {"absorption"};
  operators.tolerance = 1e-13;
  operators.maximum_iterations = 64;
  oos::SourceBatch sources;
  sources.count = 3;
  sources.initial_states = {1.0, 0.0, 0.25, 0.75, 0.0, 1.0};
  sources.direct_detection.assign(6, 0.0);
  sources.direct_losses.assign(3, 0.0);
  const auto cpu = oos::Solver::solve_cpu(operators, sources);
  const auto cuda = oos::solve_cuda(operators, sources);
  REQUIRE(cuda.iterations == cpu.iterations);
  REQUIRE(cuda.efficiency.size() == cpu.efficiency.size());
  for (std::size_t i = 0; i < cpu.efficiency.size(); ++i)
    REQUIRE(cuda.efficiency[i] == Catch::Approx(cpu.efficiency[i]).margin(1e-13));
  for (std::size_t i = 0; i < cpu.losses.size(); ++i)
    REQUIRE(cuda.losses[i] == Catch::Approx(cpu.losses[i]).margin(1e-13));
  REQUIRE(cuda.float32_efficiency_loss_upper_bound ==
          cpu.float32_efficiency_loss_upper_bound);
}

#ifdef OOS_TEST_LXE_PLUGIN_PATH
TEST_CASE("CUDA executes a factorized nonlocal LXe function block") {
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-lxe-factorized-cuda-test.h5";
  write_factorized_block(path);

  oos::OperatorSet operators;
  operators.transition = {1, 1, {0, 0}, {}, {}};
  operators.detection = {1, 1, {0, 0}, {}, {}};
  operators.losses = {1, 1, {0, 0}, {}, {}};
  operators.channel_ids = {7};
  operators.loss_names = {"absorption"};
  operators.energy_eV = 7.0;
  operators.tolerance = 1e-13;
  operators.maximum_iterations = 64;
  operators.cache_key_sha256 = "cuda-factorized-function-test";

  oos::FunctionBlock block;
  block.name = "lxe";
  block.library_path = OOS_TEST_LXE_PLUGIN_PATH;
  block.config_json =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  block.state_offset = 0;
  block.state_count = 1;
  block.egress_count = 1;
  block.intrinsic_loss_count = 1;
  block.contraction_bound = 0.4;
  block.egress_to_transition = {1, 1, {0, 1}, {0}, {0.5}};
  block.egress_to_detection = {1, 1, {0, 1}, {0}, {0.25}};
  block.egress_to_losses = {1, 1, {0, 1}, {0}, {0.25}};
  block.intrinsic_loss_columns = {0};
  operators.function_blocks.push_back(std::move(block));

  oos::SourceBatch sources;
  sources.count = 3;
  sources.initial_states = {1.0, 0.25, 0.75};
  sources.direct_detection.assign(3, 0.0);
  sources.direct_losses.assign(3, 0.0);

  const auto cpu = oos::Solver::solve_cpu(operators, sources);
  const auto cuda = oos::solve_cuda(operators, sources);
  REQUIRE(cuda.iterations == cpu.iterations);
  REQUIRE(cuda.efficiency.size() == cpu.efficiency.size());
  for (std::size_t index = 0; index < cpu.efficiency.size(); ++index)
    REQUIRE(cuda.efficiency[index] ==
            Catch::Approx(cpu.efficiency[index]).margin(1e-12));
  for (std::size_t index = 0; index < cpu.losses.size(); ++index)
    REQUIRE(cuda.losses[index] ==
            Catch::Approx(cpu.losses[index]).margin(1e-12));
  REQUIRE(cuda.float32_efficiency_loss_upper_bound ==
          cpu.float32_efficiency_loss_upper_bound);

  const auto cpu_effective =
      oos::build_effective_response_cpu(operators, 5, 1);
  const auto cuda_effective =
      oos::build_effective_response_cuda(operators, 5, 2);
  for (std::size_t index = 0;
       index < cpu_effective.state_to_detection.size(); ++index)
    REQUIRE(cuda_effective.state_to_detection[index] ==
            Catch::Approx(cpu_effective.state_to_detection[index])
                .margin(1e-12));
  for (std::size_t index = 0;
       index < cpu_effective.state_to_losses.size(); ++index)
    REQUIRE(cuda_effective.state_to_losses[index] ==
            Catch::Approx(cpu_effective.state_to_losses[index])
                .margin(1e-12));
  for (std::size_t index = 0;
       index < cpu_effective.state_unresolved.size(); ++index)
    REQUIRE(cuda_effective.state_unresolved[index] ==
            Catch::Approx(cpu_effective.state_unresolved[index])
                .margin(1e-12));
  const auto precomputed =
      oos::apply_effective_response_cuda(cuda_effective, sources);
  const auto fixed = oos::Solver::solve_cpu(
      operators, sources, oos::SolveControl{5, false});
  for (std::size_t index = 0; index < fixed.efficiency.size(); ++index)
    REQUIRE(precomputed.efficiency[index] ==
            Catch::Approx(fixed.efficiency[index]).margin(1e-12));
  for (std::size_t index = 0; index < fixed.losses.size(); ++index)
    REQUIRE(precomputed.losses[index] ==
            Catch::Approx(fixed.losses[index]).margin(1e-12));
  for (std::size_t index = 0; index < fixed.unresolved.size(); ++index)
    REQUIRE(precomputed.unresolved[index] ==
            Catch::Approx(fixed.unresolved[index]).margin(1e-12));
  std::filesystem::remove(path);
}
#endif

TEST_CASE("CUDA bounded effective response matches CPU construction and apply") {
  oos::OperatorSet operators;
  operators.transition = {2, 2, {0, 1, 2}, {1, 0}, {0.2, 0.1}};
  operators.detection = {2, 2, {0, 1, 2}, {0, 1}, {0.5, 0.6}};
  operators.losses = {2, 1, {0, 1, 2}, {0, 0}, {0.3, 0.3}};
  operators.channel_ids = {10, 20};
  operators.loss_names = {"absorption"};
  operators.cache_key_sha256 = "operator-key";
  operators.code_commit = "commit";
  operators.tolerance = 1e-13;
  const auto cpu = oos::build_effective_response_cpu(operators, 3, 2);
  const auto cuda = oos::build_effective_response_cuda(operators, 3, 2);
  for (std::size_t index = 0; index < cpu.state_to_detection.size(); ++index)
    REQUIRE(cuda.state_to_detection[index] ==
            Catch::Approx(cpu.state_to_detection[index]).margin(1e-13));
  for (std::size_t index = 0; index < cpu.state_to_losses.size(); ++index)
    REQUIRE(cuda.state_to_losses[index] ==
            Catch::Approx(cpu.state_to_losses[index]).margin(1e-13));
  oos::SourceBatch sources;
  sources.count = 2;
  sources.initial_states = {1.0, 0.0, 0.25, 0.75};
  sources.direct_detection.assign(4, 0.0);
  sources.direct_losses.assign(2, 0.0);
  oos::CudaEffectiveResponseRuntime runtime(cuda);
  const auto applied_cuda = runtime.apply(sources);
  const auto applied_cpu = oos::apply_effective_response_cpu(cpu, sources);
  for (std::size_t index = 0; index < applied_cpu.efficiency.size(); ++index)
    REQUIRE(applied_cuda.efficiency[index] ==
            Catch::Approx(applied_cpu.efficiency[index]).margin(1e-13));
}

TEST_CASE("CUDA adjoint grid likelihood matches CPU scoring") {
  const auto grid = oos::make_response_grid(
      {-1.0, 0.0, 1.0, 0.0}, {0.18, 0.02, 0.02, 0.18}, 2,
      {10, 11}, 2.0, 2.0);
  oos::HitBatch hits;
  hits.count = 2;
  hits.channels = 2;
  hits.counts = {90, 10, 10, 90};
  hits.channel_ids = {10, 11};
  const auto cpu = oos::score_response_grid_cpu(grid, hits);
  const auto cuda = oos::score_response_grid_cuda(grid, hits);
  REQUIRE(cuda.size() == cpu.size());
  for (std::size_t index = 0; index < cpu.size(); ++index)
    REQUIRE(cuda[index] == Catch::Approx(cpu[index]).margin(1e-11));
}
