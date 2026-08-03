#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "oos/hdf5_io.hpp"

TEST_CASE("geometry HDF5 preserves an independent surface basis") {
  oos::MeshData mesh;
  mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  mesh.triangles = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  mesh.surface_id = {0, 0, 1, 1};
  mesh.surface_basis_id = {7, 7, 12, 13};
  mesh.minus_domain_id.assign(4, 0);
  mesh.plus_domain_id.assign(4, -1);
  mesh.channel_id.assign(4, -1);
  mesh.triangle_transport = {1, 0, 1, 0};
  mesh.triangle_source_quadrature = {1, 1, 0, 0};
  oos::AnalyticPrimitive box;
  box.kind = oos::GeometryPrimitiveKind::box;
  box.center_mm = {1, 2, 3};
  box.parameters = {4, 5, 6, 0};
  box.surface_id = 4;
  box.surface_basis_id = 19;
  box.minus_domain_id = 0;
  box.plus_domain_id = -1;
  box.channel_id = -1;
  box.surface_element = 23;
  mesh.analytic_primitives.push_back(box);
  oos::AnalyticSurfaceElement element;
  element.primitive_index = 0;
  element.coordinates = oos::AnalyticSurfaceCoordinates::box_face_uv;
  element.bounds = {-5.0, 5.0, -6.0, 6.0, 1.0};
  element.center_mm = {4.0, 2.0, 3.0};
  element.normal = {1.0, 0.0, 0.0};
  element.area_mm2 = 120.0;
  element.surface_basis_id = 17;
  element.surface_element = 9;
  element.projected_aperture_primitive_index = 0;
  element.projected_aperture_hole_index = 3;
  mesh.analytic_surface_elements.push_back(element);
  const auto path =
      std::filesystem::temp_directory_path() / "oos-geometry-basis-test.h5";
  oos::save_geometry_hdf5(path, mesh);
  const auto restored = oos::load_geometry_hdf5(path);
  REQUIRE(restored.surface_basis_id == mesh.surface_basis_id);
  REQUIRE(restored.triangles == mesh.triangles);
  REQUIRE(restored.triangle_transport == mesh.triangle_transport);
  REQUIRE(restored.triangle_source_quadrature ==
          mesh.triangle_source_quadrature);
  REQUIRE(restored.analytic_primitives.size() == 1);
  REQUIRE(restored.analytic_primitives[0].kind ==
          oos::GeometryPrimitiveKind::box);
  REQUIRE(restored.analytic_primitives[0].parameters[2] == 6.0);
  REQUIRE(restored.analytic_primitives[0].surface_element == 23);
  REQUIRE(restored.analytic_surface_elements.size() == 1);
  REQUIRE(restored.analytic_surface_elements[0].coordinates ==
          oos::AnalyticSurfaceCoordinates::box_face_uv);
  REQUIRE(restored.analytic_surface_elements[0].area_mm2 == 120.0);
  REQUIRE(restored.analytic_surface_elements[0].surface_basis_id == 17);
  REQUIRE(restored.analytic_surface_elements[0]
              .projected_aperture_primitive_index == 0);
  REQUIRE(restored.analytic_surface_elements[0]
              .projected_aperture_hole_index == 3);
  oos::Scene scene;
  scene.mesh = restored;
  scene.apply_surface_basis(path);
  REQUIRE(scene.mesh.surface_basis_id == mesh.surface_basis_id);
  REQUIRE(scene.surface_basis_path == std::filesystem::absolute(path));
  std::filesystem::remove(path);
}

TEST_CASE("operator HDF5 round trip preserves provenance") {
  oos::OperatorSet value;
  value.transition = {1, 1, {0, 1}, {0}, {0.25}};
  value.detection = {1, 1, {0, 1}, {0}, {0.5}};
  value.losses = {1, 1, {0, 0}, {}, {}};
  value.state_labels = {"lxe/state:0"};
  value.channel_ids = {19};
  value.loss_names = {"escape"};
  value.energy_eV = 7.0;
  value.ray_origin_offset_mm = 0.002;
  value.cache_key_sha256 = "cache";
  value.scene_sha256 = "scene";
  value.geometry_sha256 = "geometry";
  value.surface_basis_sha256 = "surface-basis";
  value.dependency_lock_sha256 = "lock";
  value.code_commit = "commit";
  const auto path =
      std::filesystem::temp_directory_path() / "oos-operator-test.h5";
  oos::save_operators_hdf5(path, value);
  const auto restored = oos::load_operators_hdf5(path);
  REQUIRE(restored.channel_ids == value.channel_ids);
  REQUIRE(restored.state_labels == value.state_labels);
  REQUIRE(restored.loss_names == value.loss_names);
  REQUIRE(restored.energy_eV == value.energy_eV);
  REQUIRE(restored.ray_origin_offset_mm == value.ray_origin_offset_mm);
  REQUIRE(restored.cache_key_sha256 == value.cache_key_sha256);
  REQUIRE(restored.scene_sha256 == value.scene_sha256);
  REQUIRE(restored.geometry_sha256 == value.geometry_sha256);
  REQUIRE(restored.surface_basis_sha256 == value.surface_basis_sha256);
  REQUIRE(restored.dependency_lock_sha256 == value.dependency_lock_sha256);
  REQUIRE(restored.code_commit == value.code_commit);
  std::filesystem::remove(path);
}

TEST_CASE("operator HDF5 round trip preserves functional blocks") {
  oos::OperatorSet value;
  value.transition = {2, 2, {0, 0, 0}, {}, {}};
  value.detection = {2, 1, {0, 0, 0}, {}, {}};
  value.losses = {2, 1, {0, 0, 0}, {}, {}};
  value.state_labels = {"custom/state:0", "custom/state:1"};
  value.channel_ids = {19};
  value.loss_names = {"custom/absorption"};
  value.energy_eV = 7.0;
  oos::FunctionBlock block;
  block.name = "factorized";
  block.library_path = "/tmp/libfactorized.so";
  block.config_json = R"({"asset":"factorized.h5"})";
  block.state_offset = 0;
  block.state_count = 2;
  block.egress_count = 2;
  block.intrinsic_loss_count = 1;
  block.contraction_bound = 0.25;
  block.egress_to_transition = {2, 2, {0, 0, 0}, {}, {}};
  block.egress_to_detection =
      {2, 1, {0, 1, 2}, {0, 0}, {1.0, 1.0}};
  block.egress_to_losses = {2, 1, {0, 0, 0}, {}, {}};
  block.intrinsic_loss_columns = {0};
  value.function_blocks.push_back(block);
  const auto path =
      std::filesystem::temp_directory_path() / "oos-function-block-test.h5";
  oos::save_operators_hdf5(path, value);
  const auto restored = oos::load_operators_hdf5(path);
  REQUIRE(restored.function_blocks.size() == 1);
  const auto& actual = restored.function_blocks.front();
  REQUIRE(actual.name == block.name);
  REQUIRE(actual.library_path == block.library_path);
  REQUIRE(actual.config_json == block.config_json);
  REQUIRE(actual.state_offset == block.state_offset);
  REQUIRE(actual.state_count == block.state_count);
  REQUIRE(actual.egress_count == block.egress_count);
  REQUIRE(actual.intrinsic_loss_count == block.intrinsic_loss_count);
  REQUIRE(actual.contraction_bound == block.contraction_bound);
  REQUIRE(actual.intrinsic_loss_columns == block.intrinsic_loss_columns);
  REQUIRE(actual.egress_to_detection.data ==
          block.egress_to_detection.data);
  std::filesystem::remove(path);
}
