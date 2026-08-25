#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <cmath>

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
  std::filesystem::remove_all(root);
}
