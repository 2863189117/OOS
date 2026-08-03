#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "oos/builder.hpp"
#include "oos/hdf5_io.hpp"
#include "oos/scene.hpp"
#include "oos/solver.hpp"
#include "oos/source.hpp"
#include "oos/validation.hpp"
#ifdef OOS_HAS_CUDA
#include "oos/cuda_solver.hpp"
#endif

namespace {
void usage() {
  std::cerr
      << "usage:\n"
      << "  oos validate scene.yaml [--surface-basis selection.h5]\n"
      << "  oos build scene.yaml --cache operators.h5 "
         "[--surface-basis selection.h5] [--force]\n"
      << "  oos inspect operators.h5\n"
      << "  oos solve operators.h5 --sources source_batch.h5 "
         "--output response.h5 [--scene scene.yaml] [--device cpu|cuda]\n";
}
std::string option(int argc, char** argv, const std::string& name,
                   const std::string& fallback = "") {
  for (int i = 0; i + 1 < argc; ++i)
    if (argv[i] == name) return argv[i + 1];
  return fallback;
}
bool flag(int argc, char** argv, const std::string& name) {
  for (int i = 0; i < argc; ++i)
    if (argv[i] == name) return true;
  return false;
}
oos::Scene configured_scene(int argc, char** argv,
                            const std::filesystem::path& path) {
  oos::Scene scene = oos::Scene::load(path);
  const auto basis = option(argc, argv, "--surface-basis");
  if (!basis.empty()) scene.apply_surface_basis(basis);
  return scene;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "validate") {
      const oos::Scene scene = configured_scene(argc, argv, argv[2]);
      const auto report = oos::SceneValidator::validate(scene);
      for (const auto& issue : report.issues) {
        std::cout << (issue.severity == oos::ValidationIssue::Severity::error
                          ? "error"
                          : "warning")
                  << " [" << issue.code << "] " << issue.message << '\n';
      }
      report.throw_if_invalid();
      std::cout << "scene valid: " << scene.mesh.triangles.size()
                << " triangles, " << scene.surfaces.size() << " surfaces\n";
      return 0;
    }
    if (command == "build") {
      const std::string cache_path = option(argc, argv, "--cache");
      if (cache_path.empty()) throw std::runtime_error("--cache is required");
      const oos::Scene scene = configured_scene(argc, argv, argv[2]);
      oos::Scene configured = scene;
      const std::string energy = option(argc, argv, "--energy-eV");
      if (!energy.empty()) configured.energy_eV = std::stod(energy);
      const auto expected_key = oos::OperatorBuilder::cache_key(configured);
      if (!flag(argc, argv, "--force") &&
          std::filesystem::is_regular_file(cache_path)) {
        try {
          const auto cached = oos::load_operators_hdf5(cache_path);
          if (cached.cache_key_sha256 == expected_key) {
            std::cout << "cache hit: " << cache_path << "\ncache key: "
                      << expected_key << '\n';
            return 0;
          }
          std::cout << "cache stale: " << cache_path << '\n';
        } catch (const std::exception& exception) {
          std::cout << "cache invalid: " << cache_path << " ("
                    << exception.what() << ")\n";
        }
      }
      const auto operators = oos::OperatorBuilder::build(configured);
      const auto target = std::filesystem::path(cache_path);
      if (!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());
      auto temporary = target;
      temporary += ".tmp";
      oos::save_operators_hdf5(temporary, operators);
      std::filesystem::rename(temporary, target);
      std::cout << "built " << operators.transition.rows << " states and "
                << operators.transition.data.size()
                << " transition entries\ncache key: "
                << operators.cache_key_sha256 << '\n';
      return 0;
    }
    const oos::OperatorSet operators = oos::load_operators_hdf5(argv[2]);
    if (command == "inspect") {
      std::cout << "states: " << operators.transition.rows
                << "\nchannels: " << operators.detection.cols
                << "\nloss channels: " << operators.losses.cols
                << "\ntransition nonzeros: " << operators.transition.data.size()
                << "\nenergy (eV): " << operators.energy_eV
                << "\nray-origin offset (mm): "
                << operators.ray_origin_offset_mm
                << "\ncache key: " << operators.cache_key_sha256
                << "\ncode commit: " << operators.code_commit
                << '\n';
      return 0;
    }
    if (command == "solve") {
      const std::string source_path = option(argc, argv, "--sources");
      const std::string output_path = option(argc, argv, "--output");
      const std::string device = option(argc, argv, "--device", "cpu");
      if (source_path.empty() || output_path.empty())
        throw std::runtime_error("--sources and --output are required");
      oos::SourceBatch sources;
      if (std::filesystem::path(source_path).extension() == ".yaml") {
        const std::string scene_path = option(argc, argv, "--scene");
        if (scene_path.empty())
          throw std::runtime_error("--scene is required for sources.yaml");
        const oos::Scene scene =
            configured_scene(argc, argv, scene_path);
        sources = oos::trace_source_quadratures(
            scene, operators, oos::load_sources_yaml(source_path, scene));
      } else {
        sources = oos::load_source_batch_hdf5(
            source_path, operators.transition.rows, operators.detection.cols,
            operators.losses.cols);
      }
      oos::SolveResult result;
      if (device == "cpu") {
        result = oos::Solver::solve_cpu(operators, sources);
      } else if (device == "cuda") {
#ifdef OOS_HAS_CUDA
        result = oos::solve_cuda(operators, sources);
#else
        throw std::runtime_error("this build has no CUDA backend");
#endif
      } else {
        throw std::runtime_error("--device must be cpu or cuda");
      }
      oos::save_response_hdf5(output_path, result, operators, sources.count);
      std::cout << "solved " << sources.count << " sources in "
                << result.iterations << " iterations\n";
      if (!result.float32_efficiency_loss_upper_bound.empty())
        std::cout << "maximum float32 efficiency-loss upper bound: "
                  << *std::max_element(
                         result.float32_efficiency_loss_upper_bound.begin(),
                         result.float32_efficiency_loss_upper_bound.end())
                  << '\n';
      if (!result.source_integration_l1_error_estimate.empty())
        std::cout << "maximum source-integration L1 error estimate: "
                  << *std::max_element(
                         result.source_integration_l1_error_estimate.begin(),
                         result.source_integration_l1_error_estimate.end())
                  << '\n';
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& exception) {
    std::cerr << "oos: " << exception.what() << '\n';
    return 1;
  }
}
