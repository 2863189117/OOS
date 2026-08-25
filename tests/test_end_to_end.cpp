#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include "oos/builder.hpp"
#include "oos/hdf5_io.hpp"
#include "oos/source.hpp"
#include "oos/validation.hpp"

TEST_CASE("YAML and HDF5 scene builds and solves end to end") {
  const auto root =
      std::filesystem::temp_directory_path() / "oos-end-to-end-test";
  std::filesystem::create_directories(root);
  oos::MeshData mesh;
  mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  mesh.triangles = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  mesh.surface_id = {0, 1, 1, 1};
  mesh.minus_domain_id.assign(4, 0);
  mesh.plus_domain_id.assign(4, -1);
  mesh.channel_id = {-1, 10, 11, 12};
  oos::save_geometry_hdf5(root / "geometry.h5", mesh);
  {
    std::ofstream scene(root / "scene.yaml");
    scene << R"(schema_version: 1
energy_eV: 7
primary_domain: 0
primary_domain_seed_mm: [0.1, 0.1, 0.1]
geometry_hdf5: geometry.h5
media:
  - {id: 0, name: gas, refractive_index: 1.0, absorption_length_mm: 1000}
surfaces:
  - {id: 0, name: wall, model: lambertian, two_sided: true,
     reflectivity: 0.8}
  - {id: 1, name: detector, model: sensitive, two_sided: true,
     detection_probability: 1.0, remainder: absorb}
)";
  }
  {
    std::ofstream sources(root / "sources.yaml");
    sources << R"(schema_version: 1
sources:
  - id: center
    spatial: {type: point, position_mm: [0.1, 0.1, 0.1]}
    angular: {type: isotropic, count: 256}
)";
  }
  const auto scene = oos::Scene::load(root / "scene.yaml");
  REQUIRE(oos::SceneValidator::validate(scene).ok());
  const auto operators = oos::OperatorBuilder::build(scene);
  const oos::SourceTraceRuntime runtime(scene, operators);
  const auto sources = runtime.trace(
      oos::load_sources_yaml(root / "sources.yaml", scene));
  const auto result = oos::Solver::solve_cpu(operators, sources);
  REQUIRE(result.input_weight.at(0) > 0.999999999);
  double accounted = result.unresolved.at(0);
  for (double value : result.efficiency) accounted += value;
  for (double value : result.losses) accounted += value;
  REQUIRE(std::abs(accounted - result.input_weight.at(0)) < 1e-10);
  oos::save_response_hdf5(root / "response.h5", result, operators, 1);
  const auto file =
      H5Fopen((root / "response.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(file >= 0);
  REQUIRE(H5Lexists(file,
                    "/response/float32_efficiency_loss_upper_bound",
                    H5P_DEFAULT) > 0);
  REQUIRE(H5Lexists(file,
                    "/response/source_integration_l1_error_estimate",
                    H5P_DEFAULT) > 0);
  REQUIRE(H5Lexists(file, "/response/minimum_channel_efficiency",
                    H5P_DEFAULT) > 0);
  REQUIRE(H5Lexists(file,
                    "/response/nonnegative_pmt_probability_certified",
                    H5P_DEFAULT) > 0);
  const auto dataset =
      H5Dopen2(file, "/response/float32_efficiency_loss_upper_bound",
               H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  double bound = -1.0;
  REQUIRE(H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, &bound) >= 0);
  REQUIRE(bound == result.float32_efficiency_loss_upper_bound.at(0));
  H5Dclose(dataset);
  H5Fclose(file);
  auto invalid = result;
  REQUIRE(invalid.efficiency.size() >= 3);
  std::fill(invalid.efficiency.begin(), invalid.efficiency.end(), 0.0);
  invalid.efficiency[0] = 0.6;
  invalid.efficiency[1] = -0.1;
  invalid.efficiency[2] = 0.2;
  const auto invalid_path = root / "negative-response.h5";
  oos::save_response_hdf5(invalid_path, invalid, operators, 1);
  const auto projected_file =
      H5Fopen(invalid_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(projected_file >= 0);
  const auto projected_dataset = H5Dopen2(
      projected_file, "/response/efficiency", H5P_DEFAULT);
  REQUIRE(projected_dataset >= 0);
  std::vector<double> projected(invalid.efficiency.size());
  REQUIRE(H5Dread(projected_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, projected.data()) >= 0);
  REQUIRE(*std::min_element(projected.begin(), projected.end()) >= 0.0);
  REQUIRE(std::accumulate(projected.begin(), projected.end(), 0.0) ==
          Catch::Approx(0.7));
  REQUIRE(projected[0] == Catch::Approx(0.525));
  REQUIRE(projected[1] == 0.0);
  REQUIRE(projected[2] == Catch::Approx(0.175));
  H5Dclose(projected_dataset);

  const auto read_double = [projected_file](const char* path) {
    const auto audit = H5Dopen2(projected_file, path, H5P_DEFAULT);
    REQUIRE(audit >= 0);
    double value = 0.0;
    REQUIRE(H5Dread(audit, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &value) >= 0);
    H5Dclose(audit);
    return value;
  };
  const auto read_u64 = [projected_file](const char* path) {
    const auto audit = H5Dopen2(projected_file, path, H5P_DEFAULT);
    REQUIRE(audit >= 0);
    std::uint64_t value = 0;
    REQUIRE(H5Dread(audit, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &value) >= 0);
    H5Dclose(audit);
    return value;
  };
  const auto read_u8 = [projected_file](const char* path) {
    const auto audit = H5Dopen2(projected_file, path, H5P_DEFAULT);
    REQUIRE(audit >= 0);
    std::uint8_t value = 0;
    REQUIRE(H5Dread(audit, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &value) >= 0);
    H5Dclose(audit);
    return value;
  };
  REQUIRE(read_double("/response/raw_minimum_channel_efficiency") ==
          Catch::Approx(-0.1));
  REQUIRE(read_double("/response/minimum_channel_efficiency") == 0.0);
  REQUIRE(read_u64("/response/projected_negative_channel_count") == 1);
  REQUIRE(read_double("/response/projected_negative_efficiency_mass") ==
          Catch::Approx(0.1));
  REQUIRE(read_double("/response/nonnegative_projection_conditional_tv") ==
          Catch::Approx(1.0 / 7.0));
  REQUIRE(read_u8("/response/nonnegative_projection_applied") == 1);
  REQUIRE(read_double("/response/raw_detected_efficiency") ==
          Catch::Approx(0.7));
  REQUIRE(read_double("/response/projected_detected_efficiency") ==
          Catch::Approx(0.7));
  REQUIRE(std::abs(
              read_double("/response/nonnegative_projection_mass_error")) <
          1.0e-15);
  const auto method_dataset = H5Dopen2(
      projected_file, "/response/nonnegative_projection_method",
      H5P_DEFAULT);
  REQUIRE(method_dataset >= 0);
  const auto method_space = H5Dget_space(method_dataset);
  REQUIRE(method_space >= 0);
  const auto method_size = H5Sget_simple_extent_npoints(method_space);
  REQUIRE(method_size > 0);
  std::vector<std::uint8_t> method_bytes(
      static_cast<std::size_t>(method_size));
  REQUIRE(H5Dread(method_dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, method_bytes.data()) >= 0);
  REQUIRE(std::string(method_bytes.begin(), method_bytes.end()) ==
          "clip_negative_scale_positive_preserve_detected_mass_v1");
  H5Sclose(method_space);
  H5Dclose(method_dataset);
  H5Fclose(projected_file);

  const auto require_projection_rejected =
      [&](const std::string& name, std::vector<double> efficiency) {
        auto candidate = result;
        REQUIRE(efficiency.size() <= candidate.efficiency.size());
        std::fill(candidate.efficiency.begin(), candidate.efficiency.end(),
                  0.0);
        std::copy(efficiency.begin(), efficiency.end(),
                  candidate.efficiency.begin());
        REQUIRE_THROWS_AS(oos::save_response_hdf5(root / name, candidate,
                                                  operators, 1),
                          std::runtime_error);
      };
  require_projection_rejected(
      "nonfinite-response.h5",
      {std::numeric_limits<double>::quiet_NaN(), 0.2, 0.1});
  require_projection_rejected("zero-mass-response.h5", {0.1, -0.1, 0.0});
  require_projection_rejected("negative-mass-response.h5",
                              {0.05, -0.1, 0.0});
  require_projection_rejected(
      "overflow-response.h5",
      {std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max(), -1.0});
  require_projection_rejected(
      "rounded-unit-scale-response.h5",
      {1.0, -std::numeric_limits<double>::denorm_min(), 0.0});
  std::filesystem::remove_all(root);
}
