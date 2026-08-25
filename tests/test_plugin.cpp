#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <map>
#include <numeric>
#include <vector>

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "oos/hdf5_io.hpp"
#include "oos/builder.hpp"
#include "oos/function_operator.hpp"
#include "oos/plugin.hpp"
#include "oos/source.hpp"
#include "oos/validation.hpp"

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

void csr(hid_t file, const std::string& root, double value) {
  group(file, root.c_str());
  dataset(file, (root + "/shape").c_str(), H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{1, 1}, {2});
  dataset(file, (root + "/indptr").c_str(), H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{0, 1}, {2});
  dataset(file, (root + "/indices").c_str(), H5T_NATIVE_UINT32,
          std::vector<std::uint32_t>{0}, {1});
  dataset(file, (root + "/data").c_str(), H5T_NATIVE_DOUBLE,
          std::vector<double>{value}, {1});
}

void write_intrinsic_block(const std::filesystem::path& path,
                           double emission = 0.4) {
  const auto file =
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);
  group(file, "/nonlocal");
  group(file, "/nonlocal/egress");
  group(file, "/metadata");
  csr(file, "/nonlocal/internal_transition", 0.1);
  csr(file, "/nonlocal/emission", emission);
  csr(file, "/nonlocal/internal_losses", 0.9 - emission);
  dataset(file, "/nonlocal/egress/surface_element", H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{0}, {1});
  dataset(file, "/nonlocal/egress/barycentric", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, {1, 3});
  dataset(file, "/nonlocal/egress/side", H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{0}, {1});
  dataset(file, "/nonlocal/egress/direction_local", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0, 0.9578262852211513,
                              0.2873478855663454},
          {1, 3});
  const std::string losses = R"(["internal_absorption"])";
  dataset(file, "/metadata/loss_names_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(losses.begin(), losses.end()),
          {losses.size()});
  H5Fclose(file);
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
  dataset(file, "/nonlocal/egress/reference_axis_local", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0, 0.0, 0.0}, {1, 3});
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
      R"({"schema":"oos.nonlocal.function.v1","state_count":1,"egress_count":1,"surface_phi_bins":1,"angular_count":1,"contraction_bound":0.4,"explicit_collision_order":7})";
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

void write_complex_adjoint_block(const std::filesystem::path& path) {
  const auto file =
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);
  group(file, "/function");
  group(file, "/metadata");
  dataset(file, "/function/coefficients_real", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.18, 0.07}, {1, 1, 1, 2, 1});
  dataset(file, "/function/coefficients_imag", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0, -0.03}, {1, 1, 1, 2, 1});
  dataset(file, "/function/expected_return", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.45}, {1, 1, 1});
  dataset(file, "/function/surface_ring_area_mm2", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.7}, {1});
  dataset(file, "/function/angular_weight", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.25, 0.75}, {2});
  const std::string grid =
      R"({"position_radial_bins":1,"position_phi_bins":3,"direction_mu_bins":1,"direction_phi_bins":1})";
  const std::string generator =
      R"({"schema":"oos.nonlocal.function.v1","state_count":3,"egress_count":8,"surface_phi_bins":4,"angular_count":2,"contraction_bound":0.45,"explicit_collision_order":7})";
  dataset(file, "/metadata/phase_grid_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(grid.begin(), grid.end()),
          {grid.size()});
  dataset(file, "/metadata/generator_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(generator.begin(), generator.end()),
          {generator.size()});
  H5Fclose(file);
}

void write_joint_angular_block(const std::filesystem::path& path) {
  const auto file =
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);
  group(file, "/function");
  group(file, "/metadata");
  dataset(file, "/function/coefficients_real", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.10, 0.30, 0.02, -0.01},
          {1, 1, 1, 2, 1, 2});
  dataset(file, "/function/coefficients_imag", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0, 0.0, -0.01, 0.015},
          {1, 1, 1, 2, 1, 2});
  dataset(file, "/function/expected_return", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.4}, {1, 1, 1});
  dataset(file, "/function/surface_ring_area_mm2", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0}, {1});
  dataset(file, "/function/angular_weight", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.25, 0.75}, {2});
  const std::string grid =
      R"({"position_radial_bins":1,"position_phi_bins":3,"direction_mu_bins":1,"direction_phi_bins":1})";
  const std::string generator =
      R"({"schema":"oos.nonlocal.function.v1","state_count":3,"egress_count":8,"surface_phi_bins":4,"angular_count":2,"contraction_bound":0.4,"explicit_collision_order":7,"coefficient_layout":"joint_surface_angle_v1"})";
  dataset(file, "/metadata/phase_grid_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(grid.begin(), grid.end()),
          {grid.size()});
  dataset(file, "/metadata/generator_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(generator.begin(), generator.end()),
          {generator.size()});
  H5Fclose(file);
}

void write_ragged_factorized_block(const std::filesystem::path& path) {
  const auto file =
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);
  group(file, "/function");
  group(file, "/metadata");
  dataset(file, "/function/coefficients_real", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.15, 0.25, 0.04, -0.03},
          {1, 1, 1, 2, 2});
  dataset(file, "/function/coefficients_imag", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0, 0.0, -0.02, 0.01},
          {1, 1, 1, 2, 2});
  dataset(file, "/function/expected_return", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.4}, {1, 1, 1});
  dataset(file, "/function/surface_ring_area_mm2", H5T_NATIVE_DOUBLE,
          std::vector<double>{1.0, 1.0}, {2});
  dataset(file, "/function/surface_ring_offsets", H5T_NATIVE_UINT64,
          std::vector<std::uint64_t>{0, 2, 5}, {3});
  dataset(file, "/function/surface_phi_rad", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.0, std::acos(-1.0), 0.0,
                              2.0 * std::acos(-1.0) / 3.0,
                              4.0 * std::acos(-1.0) / 3.0},
          {5});
  dataset(file, "/function/surface_area_mm2", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0,
                              1.0 / 3.0},
          {5});
  dataset(file, "/function/angular_weight", H5T_NATIVE_DOUBLE,
          std::vector<double>{0.25, 0.75}, {2});
  const std::string grid =
      R"({"position_radial_bins":1,"position_phi_bins":3,"direction_mu_bins":1,"direction_phi_bins":1})";
  const std::string generator =
      R"({"schema":"oos.nonlocal.function.v2","surface_layout":"ragged_ring_v1","surface_point_count":5,"state_count":3,"egress_count":10,"angular_count":2,"contraction_bound":0.4,"explicit_collision_order":7})";
  dataset(file, "/metadata/phase_grid_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(grid.begin(), grid.end()),
          {grid.size()});
  dataset(file, "/metadata/generator_json", H5T_NATIVE_UINT8,
          std::vector<std::uint8_t>(generator.begin(), generator.end()),
          {generator.size()});
  H5Fclose(file);
}

void overwrite_ragged_surface_area(const std::filesystem::path& path,
                                   const std::vector<double>& values) {
  REQUIRE(values.size() == 5);
  const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const auto data =
      H5Dopen2(file, "/function/surface_area_mm2", H5P_DEFAULT);
  REQUIRE(data >= 0);
  REQUIRE(H5Dwrite(data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   values.data()) >= 0);
  H5Dclose(data);
  H5Fclose(file);
}

void overwrite_ragged_expected_return(const std::filesystem::path& path,
                                      double value) {
  const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const auto data =
      H5Dopen2(file, "/function/expected_return", H5P_DEFAULT);
  REQUIRE(data >= 0);
  REQUIRE(H5Dwrite(data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &value) >= 0);
  H5Dclose(data);
  H5Fclose(file);
}
}  // namespace

#ifdef OOS_TEST_LXE_PLUGIN_PATH
TEST_CASE("operator cache key includes intrinsic custom block contents") {
  const auto path =
      std::filesystem::temp_directory_path() / "oos-plugin-cache-key.h5";
  write_intrinsic_block(path, 0.4);
  oos::Scene scene;
  scene.energy_eV = 7.0;
  oos::SurfaceModel custom;
  custom.id = 0;
  custom.name = "lxe";
  custom.kind = oos::SurfaceKind::custom_nonlocal;
  custom.plugin_path = OOS_TEST_LXE_PLUGIN_PATH;
  custom.plugin_config_json =
      nlohmann::json{{"precomputed_block_hdf5", path.string()}}.dump();
  scene.surfaces.emplace(0, custom);
  const auto first = oos::OperatorBuilder::cache_key(scene);
  write_intrinsic_block(path, 0.3);
  const auto second = oos::OperatorBuilder::cache_key(scene);
  REQUIRE(first != second);
  std::filesystem::remove(path);
}

TEST_CASE("LXe plugin emits a native standard operator block") {
  const auto path =
      std::filesystem::temp_directory_path() / "oos-plugin-operator.h5";
  write_intrinsic_block(path);
  oos::SurfacePlugin plugin(OOS_TEST_LXE_PLUGIN_PATH);
  const std::string config =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"phase_grid", {{"position_radial_bins", 1},
                                      {"position_phi_bins", 1},
                                      {"direction_mu_bins", 1},
                                      {"direction_phi_bins", 1}}},
                     {"nonlocal_domain_id", 1},
                     {"precomputed_block_hdf5", path.string()}}
          .dump();
  plugin.validate(config, 7.0);
  REQUIRE(plugin.locality() == oos::PluginLocality::nonlocal);
  const auto result = plugin.build(config, 7.0);
  REQUIRE(result.f64.at("/nonlocal/internal_transition/data").values ==
          std::vector<double>{0.1});
  REQUIRE(result.u64.at("/nonlocal/internal_transition/shape").values ==
          std::vector<std::uint64_t>{1, 1});
  REQUIRE(result.metadata_json.find("explicit_collision_order") !=
          std::string::npos);
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  scene.media.emplace(1, oos::Medium{1, "custom", 1.1, 1000.0});
  oos::SurfaceModel custom;
  custom.id = 0;
  custom.name = "lxe";
  custom.kind = oos::SurfaceKind::custom_nonlocal;
  custom.frame_declared = true;
  custom.plugin_path = OOS_TEST_LXE_PLUGIN_PATH;
  custom.plugin_config_json =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"phase_grid", {{"position_radial_bins", 1},
                                      {"position_phi_bins", 1},
                                      {"direction_mu_bins", 1},
                                      {"direction_phi_bins", 1}}},
                     {"nonlocal_domain_id", 1},
                     {"precomputed_block_hdf5", path.string()}}
          .dump();
  scene.surfaces.emplace(0, custom);
  oos::SurfaceModel detector;
  detector.id = 1;
  detector.name = "detector";
  detector.kind = oos::SurfaceKind::sensitive;
  detector.detection_probability = 1.0;
  detector.remainder = oos::RemainderAction::absorb;
  scene.surfaces.emplace(1, detector);
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id = {1, 0, 0, 0};
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, 1);
  scene.mesh.channel_id = {19, -1, -1, -1};
  const auto imported = oos::OperatorBuilder::build(scene);
  REQUIRE(imported.channel_ids == std::vector<std::int32_t>{19});
  oos::SourceQuadrature source;
  source.id = "inside";
  source.rays.push_back(
      {{{0.1, 0.1, 0.1}, {1.0, 1.0, 1.0}},
       {1.0, 0.0, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       0});
  const oos::SourceTraceRuntime runtime(scene, imported);
  const auto batch = runtime.trace({source});
  REQUIRE(batch.initial_states.at(0) > 0.0);
  const auto solved = oos::Solver::solve_cpu(imported, batch);
  double accounted = solved.unresolved.at(0);
  for (const auto response : solved.efficiency) accounted += response;
  for (const auto loss : solved.losses) accounted += loss;
  REQUIRE(std::abs(accounted - solved.input_weight.at(0)) < 1.0e-10);
  std::filesystem::remove(path);
}

TEST_CASE("LXe factorized block executes through the function ABI") {
  const auto path =
      std::filesystem::temp_directory_path() / "oos-lxe-factorized.h5";
  write_factorized_block(path);
  const auto config =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  oos::SurfacePlugin plugin(OOS_TEST_LXE_PLUGIN_PATH);
  plugin.validate(config, 7.0);
  const auto mismatched =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 2},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  REQUIRE_THROWS_WITH(
      plugin.validate(mismatched, 7.0),
      Catch::Matchers::ContainsSubstring("does not match the function block"));
  const auto payload = plugin.build(config, 7.0);
  REQUIRE(nlohmann::json::parse(payload.metadata_json)
              .at("execution")
              .get<std::string>() == "function");
  oos::FunctionOperator function(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0);
  oos::validate_function_operator(function, 1.0e-12);
  const auto applied = function.apply_cpu(2, {1.0, 0.25});
  REQUIRE(applied.egress[0] == Catch::Approx(0.4).margin(1.0e-12));
  REQUIRE(applied.losses[0] == Catch::Approx(0.6).margin(1.0e-12));
  REQUIRE(applied.egress[1] == Catch::Approx(0.1).margin(1.0e-12));
  REQUIRE(applied.losses[1] == Catch::Approx(0.15).margin(1.0e-12));
  std::filesystem::remove(path);
}

TEST_CASE("LXe complex Fourier action and adjoint satisfy duality") {
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-lxe-complex-adjoint.h5";
  write_complex_adjoint_block(path);
  const auto config =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  oos::FunctionOperator function(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0);
  oos::validate_function_operator(function, 1.0e-12);
  std::filesystem::remove(path);
}

TEST_CASE("LXe joint surface-angle coefficients preserve direction and duality") {
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-lxe-joint-angular.h5";
  write_joint_angular_block(path);
  const auto config =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  oos::FunctionOperator function(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0);
  oos::validate_function_operator(function, 1.0e-12);

  const auto applied = function.apply_cpu(1, {1.0, 0.0, 0.0});
  REQUIRE(applied.egress.size() == 8);
  REQUIRE(std::all_of(applied.egress.begin(), applied.egress.end(),
                      [](double value) { return value >= 0.0; }));
  double angle_zero = 0.0;
  double angle_one = 0.0;
  for (std::size_t point = 0; point < 4; ++point) {
    angle_zero += applied.egress[2 * point];
    angle_one += applied.egress[2 * point + 1];
  }
  REQUIRE(angle_zero == Catch::Approx(0.10).margin(1.0e-12));
  REQUIRE(angle_one == Catch::Approx(0.30).margin(1.0e-12));
  REQUIRE(applied.losses[0] == Catch::Approx(0.60).margin(1.0e-12));
  std::filesystem::remove(path);
}

TEST_CASE("LXe v2 ragged rings preserve duality and probability") {
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-lxe-ragged-adjoint.h5";
  write_ragged_factorized_block(path);
  const auto config =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  oos::FunctionOperator function(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0);
  REQUIRE(function.descriptor().input_state_count == 3);
  REQUIRE(function.descriptor().egress_count == 10);
  oos::validate_function_operator(function, 1.0e-12);

  const auto applied = function.apply_cpu(
      3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  for (std::uint64_t row = 0; row < 3; ++row) {
    const auto begin = applied.egress.begin() + row * 10;
    const double returned = std::accumulate(begin, begin + 10, 0.0);
    REQUIRE(returned == Catch::Approx(0.4).margin(1.0e-12));
    REQUIRE(applied.losses[row] == Catch::Approx(0.6).margin(1.0e-12));
  }
  std::filesystem::remove(path);
}

TEST_CASE("LXe v2 rejects nonconservative ragged quadrature") {
  const auto path = std::filesystem::temp_directory_path() /
                    "oos-lxe-ragged-invalid.h5";
  write_ragged_factorized_block(path);
  const auto config =
      nlohmann::json{{"geometry", "finite_cylinder"},
                     {"explicit_collision_order", 7},
                     {"factorized_block_hdf5", path.string()}}
          .dump();
  SECTION("ring area mismatch") {
    overwrite_ragged_surface_area(
        path, {0.6, 0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
    REQUIRE_THROWS(
        oos::FunctionOperator(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0));
  }
  SECTION("retained Fourier mode alias") {
    overwrite_ragged_surface_area(
        path, {0.6, 0.4, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
    REQUIRE_THROWS(
        oos::FunctionOperator(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0));
  }
  SECTION("m0 return mismatch") {
    overwrite_ragged_expected_return(path, 0.41);
    REQUIRE_THROWS(
        oos::FunctionOperator(OOS_TEST_LXE_PLUGIN_PATH, config, 7.0));
  }
  std::filesystem::remove(path);
}
#endif

TEST_CASE("local custom surface is composed with generic states") {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  oos::SurfaceModel custom;
  custom.id = 0;
  custom.name = "coating";
  custom.kind = oos::SurfaceKind::custom_local;
  custom.frame_declared = true;
  custom.plugin_path = OOS_TEST_LOCAL_PLUGIN_PATH;
  custom.plugin_config_json =
      nlohmann::json{{"detection", 0.5}, {"absorption", 0.5}}.dump();
  scene.surfaces.emplace(0, custom);
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id.assign(4, 0);
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, -1);
  scene.mesh.channel_id = {10, 11, 12, 13};
  const auto report = oos::SceneValidator::validate(scene);
  REQUIRE(report.ok());
  const auto operators = oos::OperatorBuilder::build(scene);
  REQUIRE(operators.transition.rows == 4);
  REQUIRE(operators.channel_ids == std::vector<std::int32_t>{10, 11, 12, 13});
  for (std::uint64_t row = 0; row < operators.transition.rows; ++row) {
    double terminal = 0.0;
    for (auto i = operators.detection.indptr[row];
         i < operators.detection.indptr[row + 1]; ++i)
      terminal += operators.detection.data[i];
    for (auto i = operators.losses.indptr[row];
         i < operators.losses.indptr[row + 1]; ++i)
      terminal += operators.losses.data[i];
    REQUIRE(std::abs(terminal - 1.0) < 1.0e-12);
  }
}

TEST_CASE("nonlocal egress rays are mapped by the core geometry") {
  oos::Scene scene;
  scene.energy_eV = 7.0;
  scene.primary_domain = 0;
  scene.primary_domain_seed_mm = {0.1, 0.1, 0.1};
  scene.media.emplace(0, oos::Medium{0, "inside", 1.0, 1000.0});
  scene.media.emplace(1, oos::Medium{1, "custom", 1.1, 1000.0});
  oos::SurfaceModel local;
  local.id = 0;
  local.name = "local";
  local.kind = oos::SurfaceKind::custom_local;
  local.frame_declared = true;
  local.plugin_path = OOS_TEST_LOCAL_PLUGIN_PATH;
  local.plugin_config_json =
      nlohmann::json{{"detection", 0.5}, {"absorption", 0.5}}.dump();
  scene.surfaces.emplace(0, local);
  oos::SurfaceModel nonlocal;
  nonlocal.id = 1;
  nonlocal.name = "nonlocal";
  nonlocal.kind = oos::SurfaceKind::custom_nonlocal;
  nonlocal.frame_declared = true;
  nonlocal.plugin_path = OOS_TEST_NONLOCAL_PLUGIN_PATH;
  nonlocal.plugin_config_json =
      nlohmann::json{{"nonlocal_domain_id", 1}}.dump();
  scene.surfaces.emplace(1, nonlocal);
  scene.mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  scene.mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  scene.mesh.surface_id = {0, 1, 1, 1};
  scene.mesh.minus_domain_id.assign(4, 0);
  scene.mesh.plus_domain_id.assign(4, 1);
  scene.mesh.channel_id = {10, -1, -1, -1};
  REQUIRE(oos::SceneValidator::validate(scene).ok());
  const auto operators = oos::OperatorBuilder::build(scene);
  REQUIRE(operators.transition.rows == 2);
  REQUIRE(operators.state_labels ==
          std::vector<std::string>{"local/primitive:0",
                                   "nonlocal/state:0"});
  REQUIRE(operators.channel_ids == std::vector<std::int32_t>{10});
  const std::uint64_t plugin_row = 1;
  std::map<std::uint32_t, double> transition;
  for (auto entry = operators.transition.indptr[plugin_row];
       entry < operators.transition.indptr[plugin_row + 1]; ++entry)
    transition[operators.transition.indices[entry]] =
        operators.transition.data[entry];
  REQUIRE(std::abs(transition.at(1) - 0.1) < 1.0e-12);
  double detected = 0.0;
  for (auto entry = operators.detection.indptr[plugin_row];
       entry < operators.detection.indptr[plugin_row + 1]; ++entry)
    detected += operators.detection.data[entry];
  REQUIRE(detected < 0.2);
  REQUIRE(detected > 0.199);
  double accounted = 0.0;
  for (auto entry = operators.transition.indptr[plugin_row];
       entry < operators.transition.indptr[plugin_row + 1]; ++entry)
    accounted += operators.transition.data[entry];
  for (auto entry = operators.detection.indptr[plugin_row];
       entry < operators.detection.indptr[plugin_row + 1]; ++entry)
    accounted += operators.detection.data[entry];
  for (auto entry = operators.losses.indptr[plugin_row];
       entry < operators.losses.indptr[plugin_row + 1]; ++entry)
    accounted += operators.losses.data[entry];
  REQUIRE(std::abs(accounted - 1.0) < 1.0e-12);
  REQUIRE(operators.loss_names.back() ==
          "nonlocal/internal_absorption");

  // A nonlocal egress element may be owned by an exact analytic primitive.
  // Its stable surface-local element ID then replaces any dependency on the
  // validation triangulation.
  oos::Scene analytic_egress_scene = scene;
  oos::AnalyticPrimitive interface_disk;
  interface_disk.kind = oos::GeometryPrimitiveKind::disk;
  interface_disk.center_mm = {0.0, 0.0, 0.0};
  interface_disk.parameters = {1.0, 0.0, 0.0, 0.0};
  interface_disk.normal_sign = -1.0;
  interface_disk.surface_id = 1;
  interface_disk.minus_domain_id = 0;
  interface_disk.plus_domain_id = 1;
  analytic_egress_scene.mesh.analytic_primitives.push_back(interface_disk);
  oos::AnalyticSurfaceElement interface_element;
  interface_element.primitive_index = 0;
  interface_element.coordinates =
      oos::AnalyticSurfaceCoordinates::annulus_r2_phi;
  interface_element.bounds = {0.0, 1.0, 0.0, 2.0 * std::acos(-1.0), 0.0};
  interface_element.center_mm = {0.1, 0.1, 0.0};
  interface_element.normal = {0.0, 0.0, -1.0};
  interface_element.area_mm2 = std::acos(-1.0);
  interface_element.surface_element = 0;
  interface_element.source_quadrature = false;
  analytic_egress_scene.mesh.analytic_surface_elements.push_back(
      interface_element);
  const auto analytic_egress =
      oos::OperatorBuilder::build(analytic_egress_scene);
  REQUIRE(analytic_egress.transition.rows == operators.transition.rows);
  REQUIRE(analytic_egress.channel_ids == operators.channel_ids);

  // Surface-element identifiers are local to a nonlocal surface and must
  // resolve uniquely.  The builder indexes this lookup once, but preserves
  // the previous ambiguity error when a referenced identifier is duplicated.
  oos::Scene ambiguous_egress_scene = analytic_egress_scene;
  ambiguous_egress_scene.mesh.analytic_surface_elements.push_back(
      interface_element);
  REQUIRE_THROWS_WITH(
      oos::OperatorBuilder::build(ambiguous_egress_scene),
      Catch::Matchers::ContainsSubstring(
          "nonlocal egress surface element is ambiguous"));

  oos::Scene functional_scene = scene;
  functional_scene.surfaces.at(1).plugin_config_json =
      nlohmann::json{{"nonlocal_domain_id", 1}, {"functional", true}}.dump();
  const auto functional = oos::OperatorBuilder::build(functional_scene);
  REQUIRE(functional.function_blocks.size() == 1);
  REQUIRE(functional.function_blocks.front().state_offset == plugin_row);
  REQUIRE(functional.function_blocks.front().state_count == 1);
  REQUIRE(functional.function_blocks.front().egress_count == 1);
  REQUIRE(functional.transition.indptr[plugin_row] ==
          functional.transition.indptr[plugin_row + 1]);
  oos::SourceBatch explicit_source{
      1, {0.0, 1.0}, std::vector<double>(operators.detection.cols, 0.0),
      std::vector<double>(operators.losses.cols, 0.0)};
  oos::SourceBatch functional_source{
      1, {0.0, 1.0}, std::vector<double>(functional.detection.cols, 0.0),
      std::vector<double>(functional.losses.cols, 0.0)};
  const auto explicit_result =
      oos::Solver::solve_cpu(operators, explicit_source);
  const auto functional_result =
      oos::Solver::solve_cpu(functional, functional_source);
  REQUIRE(functional_result.efficiency.size() ==
          explicit_result.efficiency.size());
  for (std::size_t index = 0;
       index < explicit_result.efficiency.size(); ++index)
    REQUIRE(std::abs(functional_result.efficiency[index] -
                     explicit_result.efficiency[index]) < 1.0e-12);
  for (std::size_t index = 0; index < explicit_result.losses.size(); ++index)
    REQUIRE(std::abs(functional_result.losses[index] -
                     explicit_result.losses[index]) < 1.0e-12);

  oos::Scene coupled = scene;
  coupled.surfaces.at(1).plugin_config_json =
      nlohmann::json{{"nonlocal_domain_id", 1},
                     {"emit_forbidden_channel", true}}
          .dump();
  REQUIRE_THROWS_WITH(
      oos::OperatorBuilder::build(coupled),
      Catch::Matchers::ContainsSubstring("geometry-coupled field"));

  // The plugin payload is unchanged after a rigid placement change.  Only the
  // core geometry mapping is rebuilt, and the assembled response is invariant.
  oos::Scene transformed = scene;
  const auto move = [](const oos::Vec3& point) {
    return oos::Vec3{4.0 - point.y, -2.0 + point.x, 3.0 + point.z};
  };
  for (auto& vertex : transformed.mesh.vertices) vertex = move(vertex);
  transformed.primary_domain_seed_mm = move(scene.primary_domain_seed_mm);
  const auto rotate = [](const oos::Vec3& vector) {
    return oos::Vec3{-vector.y, vector.x, vector.z};
  };
  for (auto& [id, surface] : transformed.surfaces) {
    (void)id;
    if (!surface.frame_declared) continue;
    surface.frame_origin_mm = move(surface.frame_origin_mm);
    surface.frame_x = rotate(surface.frame_x);
    surface.frame_y = rotate(surface.frame_y);
    surface.frame_z = rotate(surface.frame_z);
  }
  const auto relocated = oos::OperatorBuilder::build(transformed);
  REQUIRE(relocated.transition.indptr == operators.transition.indptr);
  REQUIRE(relocated.transition.indices == operators.transition.indices);
  REQUIRE(relocated.detection.indptr == operators.detection.indptr);
  REQUIRE(relocated.detection.indices == operators.detection.indices);
  REQUIRE(relocated.losses.indptr == operators.losses.indptr);
  REQUIRE(relocated.losses.indices == operators.losses.indices);
  const auto compare_plugin_row = [plugin_row](const oos::CsrMatrix& first,
                                                const oos::CsrMatrix& second) {
    REQUIRE(first.indptr[plugin_row + 1] - first.indptr[plugin_row] ==
            second.indptr[plugin_row + 1] - second.indptr[plugin_row]);
    for (auto entry = first.indptr[plugin_row],
              other = second.indptr[plugin_row];
         entry < first.indptr[plugin_row + 1]; ++entry, ++other) {
      REQUIRE(first.indices[entry] == second.indices[other]);
      REQUIRE(std::abs(first.data[entry] - second.data[other]) < 1.0e-10);
    }
  };
  compare_plugin_row(operators.transition, relocated.transition);
  compare_plugin_row(operators.detection, relocated.detection);
  compare_plugin_row(operators.losses, relocated.losses);
}
