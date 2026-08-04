#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "oos/cuda_solver.hpp"
#include "oos/effective_response.hpp"
#include "oos/hdf5_io.hpp"
#include "oos/scene.hpp"
#include "oos/source.hpp"

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  oos-efficiency precompute operators.h5 --output effective.h5 "
         "[--cycles 7] [--batch-size 64] [--device cpu|cuda]\n"
      << "  oos-efficiency calculate operators.h5 --sources sources.yaml|"
         "source_batch.h5 --output response.h5 [--scene scene.yaml] "
         "[--surface-basis selection.h5] [--precomputed effective.h5] "
         "[--device cpu|cuda] [--direct]\n";
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
  auto scene = oos::Scene::load(path);
  const auto basis = option(argc, argv, "--surface-basis");
  if (!basis.empty()) scene.apply_surface_basis(basis);
  return scene;
}

oos::SourceBatch sources_for(
    int argc, char** argv, const oos::OperatorSet& operators,
    const std::filesystem::path& source_path) {
  if (source_path.extension() != ".yaml")
    return oos::load_source_batch_hdf5(
        source_path, operators.transition.rows, operators.detection.cols,
        operators.losses.cols);
  const auto scene_path = option(argc, argv, "--scene");
  if (scene_path.empty())
    throw std::runtime_error("--scene is required for sources.yaml");
  const auto scene = configured_scene(argc, argv, scene_path);
  return oos::trace_source_quadratures(
      scene, operators, oos::load_sources_yaml(source_path, scene));
}

oos::EffectiveResponse build_response(const oos::OperatorSet& operators,
                                      const std::string& device,
                                      std::uint32_t cycles,
                                      std::uint64_t batch_size) {
  if (device == "cpu")
    return oos::build_effective_response_cpu(operators, cycles, batch_size);
  if (device == "cuda") {
#ifdef OOS_HAS_CUDA
    return oos::build_effective_response_cuda(operators, cycles, batch_size);
#else
    throw std::runtime_error("this build has no CUDA backend");
#endif
  }
  throw std::runtime_error("--device must be cpu or cuda");
}

oos::SolveResult apply_response(const oos::EffectiveResponse& response,
                                const oos::SourceBatch& sources,
                                const std::string& device) {
  if (device == "cpu")
    return oos::apply_effective_response_cpu(response, sources);
  if (device == "cuda") {
#ifdef OOS_HAS_CUDA
    return oos::apply_effective_response_cuda(response, sources);
#else
    throw std::runtime_error("this build has no CUDA backend");
#endif
  }
  throw std::runtime_error("--device must be cpu or cuda");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    const std::filesystem::path operator_path = argv[2];
    const auto operators = oos::load_operators_hdf5(operator_path);
    const auto device = option(argc, argv, "--device", "cuda");
    if (command == "precompute") {
      const auto output = option(argc, argv, "--output");
      if (output.empty()) throw std::runtime_error("--output is required");
      const auto cycles =
          static_cast<std::uint32_t>(
              std::stoul(option(argc, argv, "--cycles", "7")));
      const auto batch_size =
          static_cast<std::uint64_t>(
              std::stoull(option(argc, argv, "--batch-size", "64")));
      auto response =
          build_response(operators, device, cycles, batch_size);
      auto temporary = std::filesystem::path(output);
      temporary += ".tmp";
      oos::save_effective_response_hdf5(temporary, response);
      std::filesystem::rename(temporary, output);
      std::cout << "precomputed " << response.states << " states x "
                << response.channels << " channels for " << response.cycles
                << " cycles using the adjoint solver\n";
      return 0;
    }
    if (command == "calculate") {
      const auto source_path = option(argc, argv, "--sources");
      const auto output_path = option(argc, argv, "--output");
      if (source_path.empty() || output_path.empty())
        throw std::runtime_error("--sources and --output are required");
      const auto sources =
          sources_for(argc, argv, operators, source_path);
      oos::SolveResult result;
      if (flag(argc, argv, "--direct")) {
        if (device == "cpu")
          result = oos::Solver::solve_cpu(operators, sources);
        else {
#ifdef OOS_HAS_CUDA
          result = oos::solve_cuda(operators, sources);
#else
          throw std::runtime_error("this build has no CUDA backend");
#endif
        }
      } else {
        auto effective_path = option(argc, argv, "--precomputed");
        if (effective_path.empty())
          effective_path = operator_path.string() + ".seven-cycle.h5";
        bool rebuild =
            !std::filesystem::is_regular_file(effective_path);
        if (!rebuild) {
          try {
            const auto cached =
                oos::load_effective_response_hdf5(effective_path);
            rebuild = cached.operator_cache_key_sha256 !=
                      operators.cache_key_sha256;
            if (rebuild)
              std::cout << "effective response cache stale: "
                        << effective_path << '\n';
          } catch (const std::exception& exception) {
            rebuild = true;
            std::cout << "effective response cache invalid: "
                      << effective_path << " (" << exception.what() << ")\n";
          }
        }
        if (rebuild) {
          auto response = build_response(operators, device, 7, 64);
          auto temporary = std::filesystem::path(effective_path);
          temporary += ".tmp";
          oos::save_effective_response_hdf5(temporary, response);
          std::filesystem::rename(temporary, effective_path);
        }
        const auto response =
            oos::load_effective_response_hdf5(effective_path);
        if (response.operator_cache_key_sha256 !=
            operators.cache_key_sha256)
          throw std::runtime_error(
              "precomputed response does not match operators.h5");
        result = apply_response(response, sources, device);
      }
      oos::save_response_hdf5(output_path, result, operators, sources.count);
      std::cout << "calculated " << sources.count << " sources using "
                << result.backend << "\n";
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& exception) {
    std::cerr << "oos-efficiency: " << exception.what() << '\n';
    return 1;
  }
}
