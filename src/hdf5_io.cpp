#include "oos/hdf5_io.hpp"
#include "oos/regression.hpp"

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace oos {
namespace {

class H5Handle {
 public:
  H5Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {
    if (id_ < 0) throw std::runtime_error("HDF5 operation failed");
  }
  ~H5Handle() { closer_(id_); }
  operator hid_t() const { return id_; }

 private:
  hid_t id_;
  herr_t (*closer_)(hid_t);
};

template <typename T>
hid_t native_type();
template <>
hid_t native_type<double>() { return H5T_NATIVE_DOUBLE; }
template <>
hid_t native_type<float>() { return H5T_NATIVE_FLOAT; }
template <>
hid_t native_type<std::uint32_t>() { return H5T_NATIVE_UINT32; }
template <>
hid_t native_type<std::uint64_t>() { return H5T_NATIVE_UINT64; }
template <>
hid_t native_type<std::int32_t>() { return H5T_NATIVE_INT32; }
template <>
hid_t native_type<std::uint8_t>() { return H5T_NATIVE_UINT8; }

template <typename T>
std::pair<std::vector<T>, std::vector<hsize_t>> read_dataset(
    hid_t file, const char* path) {
  H5Handle dataset(H5Dopen2(file, path, H5P_DEFAULT), H5Dclose);
  H5Handle space(H5Dget_space(dataset), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0) throw std::runtime_error("invalid HDF5 dataspace");
  std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
  H5Sget_simple_extent_dims(space, dimensions.data(), nullptr);
  std::size_t count = 1;
  for (hsize_t dimension : dimensions) count *= dimension;
  std::vector<T> values(count);
  if (H5Dread(dataset, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
              values.data()) < 0) {
    throw std::runtime_error(std::string("cannot read ") + path);
  }
  return {std::move(values), std::move(dimensions)};
}

template <typename T>
void write_dataset(hid_t file, const char* path, const std::vector<T>& values,
                   const std::vector<hsize_t>& dimensions) {
  H5Handle space(H5Screate_simple(static_cast<int>(dimensions.size()),
                                  dimensions.data(), nullptr),
                 H5Sclose);
  H5Handle dataset(H5Dcreate2(file, path, native_type<T>(), space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT),
                   H5Dclose);
  if (H5Dwrite(dataset, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
               values.data()) < 0) {
    throw std::runtime_error(std::string("cannot write ") + path);
  }
}

CsrMatrix read_csr(hid_t file, const std::string& root) {
  auto [shape, shape_dims] =
      read_dataset<std::uint64_t>(file, (root + "/shape").c_str());
  if (shape_dims != std::vector<hsize_t>{2}) {
    throw std::runtime_error(root + "/shape must contain two values");
  }
  CsrMatrix matrix;
  matrix.rows = shape[0];
  matrix.cols = shape[1];
  matrix.indptr =
      read_dataset<std::uint64_t>(file, (root + "/indptr").c_str()).first;
  matrix.indices =
      read_dataset<std::uint32_t>(file, (root + "/indices").c_str()).first;
  matrix.data =
      read_dataset<double>(file, (root + "/data").c_str()).first;
  return matrix;
}

void create_group(hid_t file, const char* path) {
  H5Handle group(
      H5Gcreate2(file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
}

void write_csr(hid_t file, const std::string& root, const CsrMatrix& matrix) {
  create_group(file, root.c_str());
  write_dataset(file, (root + "/shape").c_str(),
                std::vector<std::uint64_t>{matrix.rows, matrix.cols}, {2});
  write_dataset(file, (root + "/indptr").c_str(), matrix.indptr,
                {matrix.indptr.size()});
  write_dataset(file, (root + "/indices").c_str(), matrix.indices,
                {matrix.indices.size()});
  write_dataset(file, (root + "/data").c_str(), matrix.data,
                {matrix.data.size()});
}

bool exists(hid_t file, const std::string& path) {
  htri_t result = 0;
  H5E_BEGIN_TRY {
    result = H5Lexists(file, path.c_str(), H5P_DEFAULT);
  }
  H5E_END_TRY;
  return result > 0;
}

void write_string(hid_t file, const std::string& path,
                  const std::string& value) {
  const std::string stored = value.empty() ? "unknown" : value;
  write_dataset(file, path.c_str(),
                std::vector<std::uint8_t>(stored.begin(), stored.end()),
                {stored.size()});
}

std::string read_string(hid_t file, const std::string& path) {
  if (!exists(file, path)) return {};
  const auto bytes = read_dataset<std::uint8_t>(file, path.c_str()).first;
  return {bytes.begin(), bytes.end()};
}

}  // namespace

MeshData load_geometry_hdf5(const std::filesystem::path& path) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  MeshData mesh;
  auto [vertices, vdims] =
      read_dataset<double>(file, "/geometry/vertices");
  auto [triangles, tdims] =
      read_dataset<std::uint32_t>(file, "/geometry/triangles");
  if (vdims.size() != 2 || vdims[1] != 3 || tdims.size() != 2 ||
      tdims[1] != 3) {
    throw std::runtime_error("vertices and triangles must have shape [N,3]");
  }
  mesh.vertices.resize(vdims[0]);
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    mesh.vertices[i] = {vertices[3 * i], vertices[3 * i + 1],
                        vertices[3 * i + 2]};
  }
  mesh.triangles.resize(tdims[0]);
  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    mesh.triangles[i] = {triangles[3 * i], triangles[3 * i + 1],
                         triangles[3 * i + 2]};
  }
  mesh.surface_id =
      read_dataset<std::uint32_t>(file, "/geometry/surface_id").first;
  mesh.surface_basis_id =
      read_dataset<std::uint32_t>(file, "/geometry/surface_basis_id").first;
  mesh.minus_domain_id =
      read_dataset<std::int32_t>(file, "/geometry/minus_domain_id").first;
  mesh.plus_domain_id =
      read_dataset<std::int32_t>(file, "/geometry/plus_domain_id").first;
  mesh.channel_id =
      read_dataset<std::int32_t>(file, "/geometry/channel_id").first;
  mesh.triangle_transport =
      read_dataset<std::uint8_t>(file, "/geometry/triangle_transport").first;
  mesh.triangle_source_quadrature =
      read_dataset<std::uint8_t>(
          file, "/geometry/triangle_source_quadrature").first;
  mesh.triangle_source_analytic_primitive =
      read_dataset<std::uint32_t>(
          file, "/geometry/triangle_source_analytic_primitive").first;
  if (exists(file, "/analytic/kind")) {
    auto [kind, kind_dims] =
        read_dataset<std::uint8_t>(file, "/analytic/kind");
    auto [center, center_dims] =
        read_dataset<double>(file, "/analytic/center_mm");
    auto [axis_x, axis_x_dims] =
        read_dataset<double>(file, "/analytic/axis_x");
    auto [axis_y, axis_y_dims] =
        read_dataset<double>(file, "/analytic/axis_y");
    auto [axis_z, axis_z_dims] =
        read_dataset<double>(file, "/analytic/axis_z");
    auto [parameters, parameter_dims] =
        read_dataset<double>(file, "/analytic/parameters");
    const auto normal_sign =
        read_dataset<double>(file, "/analytic/normal_sign").first;
    const auto surface_id =
        read_dataset<std::uint32_t>(file, "/analytic/surface_id").first;
    const auto surface_basis_id =
        read_dataset<std::uint32_t>(
            file, "/analytic/surface_basis_id").first;
    const auto minus_domain_id =
        read_dataset<std::int32_t>(
            file, "/analytic/minus_domain_id").first;
    const auto plus_domain_id =
        read_dataset<std::int32_t>(
            file, "/analytic/plus_domain_id").first;
    const auto channel_id =
        read_dataset<std::int32_t>(file, "/analytic/channel_id").first;
    const auto surface_element =
        read_dataset<std::uint64_t>(
            file, "/analytic/surface_element").first;
    const auto source_integral =
        read_dataset<std::uint8_t>(
            file, "/analytic/source_integral").first;
    const auto hole_offset =
        read_dataset<std::uint64_t>(
            file, "/analytic/hole_offset").first;
    auto [hole_center, hole_center_dims] =
        read_dataset<double>(file, "/analytic/hole_center_uv_mm");
    const auto hole_radius =
        read_dataset<double>(file, "/analytic/hole_radius_mm").first;
    const std::size_t count = kind.size();
    const auto vector_count_ok = [count](std::size_t size) {
      return size == count;
    };
    if (kind_dims != std::vector<hsize_t>{count} ||
        center_dims != std::vector<hsize_t>{count, 3} ||
        axis_x_dims != std::vector<hsize_t>{count, 3} ||
        axis_y_dims != std::vector<hsize_t>{count, 3} ||
        axis_z_dims != std::vector<hsize_t>{count, 3} ||
        parameter_dims != std::vector<hsize_t>{count, 4} ||
        !vector_count_ok(normal_sign.size()) ||
        !vector_count_ok(surface_id.size()) ||
        !vector_count_ok(surface_basis_id.size()) ||
        !vector_count_ok(minus_domain_id.size()) ||
        !vector_count_ok(plus_domain_id.size()) ||
        !vector_count_ok(channel_id.size()) ||
        !vector_count_ok(surface_element.size()) ||
        !vector_count_ok(source_integral.size()) ||
        hole_offset.size() != count + 1 ||
        hole_center_dims != std::vector<hsize_t>{hole_radius.size(), 2} ||
        hole_offset.front() != 0 ||
        hole_offset.back() != hole_radius.size()) {
      throw std::runtime_error("analytic geometry datasets have invalid shapes");
    }
    mesh.analytic_primitives.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (kind[i] <
              static_cast<std::uint8_t>(GeometryPrimitiveKind::disk) ||
          kind[i] >
              static_cast<std::uint8_t>(
                  GeometryPrimitiveKind::perforated_disk))
        throw std::runtime_error("analytic geometry has an unknown kind");
      auto& primitive = mesh.analytic_primitives[i];
      primitive.kind = static_cast<GeometryPrimitiveKind>(kind[i]);
      primitive.center_mm =
          {center[3 * i], center[3 * i + 1], center[3 * i + 2]};
      primitive.axis_x =
          {axis_x[3 * i], axis_x[3 * i + 1], axis_x[3 * i + 2]};
      primitive.axis_y =
          {axis_y[3 * i], axis_y[3 * i + 1], axis_y[3 * i + 2]};
      primitive.axis_z =
          {axis_z[3 * i], axis_z[3 * i + 1], axis_z[3 * i + 2]};
      std::copy_n(parameters.begin() + 4 * i, 4,
                  primitive.parameters.begin());
      primitive.normal_sign = normal_sign[i];
      primitive.surface_id = surface_id[i];
      primitive.surface_basis_id = surface_basis_id[i];
      primitive.minus_domain_id = minus_domain_id[i];
      primitive.plus_domain_id = plus_domain_id[i];
      primitive.channel_id = channel_id[i];
      primitive.surface_element = surface_element[i];
      if (source_integral[i] > static_cast<std::uint8_t>(
                                   AnalyticSourceIntegral::directional_disk))
        throw std::runtime_error(
            "analytic primitive has an unknown source integral");
      primitive.source_integral =
          static_cast<AnalyticSourceIntegral>(source_integral[i]);
      for (std::uint64_t hole = hole_offset[i];
           hole < hole_offset[i + 1]; ++hole)
        primitive.holes.push_back(
            {{hole_center[2 * hole], hole_center[2 * hole + 1]},
             hole_radius[hole]});
    }
  }
  if (exists(file, "/analytic/elements/primitive_index")) {
    const auto primitive_index =
        read_dataset<std::uint32_t>(
            file, "/analytic/elements/primitive_index").first;
    const auto coordinates =
        read_dataset<std::uint8_t>(
            file, "/analytic/elements/coordinates").first;
    auto [bounds, bounds_dims] =
        read_dataset<double>(file, "/analytic/elements/bounds");
    auto [center, center_dims] =
        read_dataset<double>(file, "/analytic/elements/center_mm");
    auto [normal, normal_dims] =
        read_dataset<double>(file, "/analytic/elements/normal");
    const auto area =
        read_dataset<double>(file, "/analytic/elements/area_mm2").first;
    const auto basis =
        read_dataset<std::uint32_t>(
            file, "/analytic/elements/surface_basis_id").first;
    const auto surface_element =
        read_dataset<std::uint64_t>(
            file, "/analytic/elements/surface_element").first;
    const std::size_t count = primitive_index.size();
    const auto source_quadrature =
        read_dataset<std::uint8_t>(
            file, "/analytic/elements/source_quadrature").first;
    const auto source_visibility =
        read_dataset<std::uint8_t>(
            file, "/analytic/elements/source_visibility").first;
    const auto aperture_primitive =
        read_dataset<std::uint32_t>(
            file,
            "/analytic/elements/projected_aperture_primitive_index")
            .first;
    const auto aperture_hole =
        read_dataset<std::uint32_t>(
            file, "/analytic/elements/projected_aperture_hole_index")
            .first;
    if (coordinates.size() != count ||
        bounds_dims != std::vector<hsize_t>{count, 5} ||
        center_dims != std::vector<hsize_t>{count, 3} ||
        normal_dims != std::vector<hsize_t>{count, 3} ||
        area.size() != count || basis.size() != count ||
        surface_element.size() != count ||
        source_quadrature.size() != count ||
        source_visibility.size() != count ||
        aperture_primitive.size() != count ||
        aperture_hole.size() != count)
      throw std::runtime_error(
          "analytic surface-element datasets have invalid shapes");
    mesh.analytic_surface_elements.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (coordinates[i] >
          static_cast<std::uint8_t>(
              AnalyticSurfaceCoordinates::box_face_uv))
        throw std::runtime_error(
            "analytic surface element has unknown coordinates");
      auto& element = mesh.analytic_surface_elements[i];
      element.primitive_index = primitive_index[i];
      element.coordinates =
          static_cast<AnalyticSurfaceCoordinates>(coordinates[i]);
      std::copy_n(bounds.begin() + 5 * i, 5, element.bounds.begin());
      element.center_mm =
          {center[3 * i], center[3 * i + 1], center[3 * i + 2]};
      element.normal =
          {normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]};
      element.area_mm2 = area[i];
      element.surface_basis_id = basis[i];
      element.surface_element = surface_element[i];
      element.source_quadrature = source_quadrature[i] != 0;
      if (source_visibility[i] > static_cast<std::uint8_t>(
                                     AnalyticSourceVisibility::projected_aperture))
        throw std::runtime_error(
            "analytic surface element has unknown source visibility");
      element.source_visibility =
          static_cast<AnalyticSourceVisibility>(source_visibility[i]);
      element.projected_aperture_primitive_index =
          aperture_primitive[i];
      element.projected_aperture_hole_index = aperture_hole[i];
    }
  }
  return mesh;
}

std::vector<std::uint32_t> load_surface_basis_id_hdf5(
    const std::filesystem::path& path, std::uint64_t triangle_count) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  const std::string dataset =
      exists(file, "/surface_basis_selection/triangle_basis_id")
          ? "/surface_basis_selection/triangle_basis_id"
          : "/geometry/surface_basis_id";
  if (!exists(file, dataset))
    throw std::runtime_error(
        "surface basis file lacks triangle_basis_id");
  auto [values, dimensions] =
      read_dataset<std::uint32_t>(file, dataset.c_str());
  if (dimensions != std::vector<hsize_t>{triangle_count})
    throw std::runtime_error(
        "surface basis triangle count differs from geometry");
  return values;
}

void save_geometry_hdf5(const std::filesystem::path& path,
                        const MeshData& mesh) {
  const auto triangle_count = mesh.triangles.size();
  if (mesh.surface_id.size() != triangle_count ||
      (!mesh.surface_basis_id.empty() &&
       mesh.surface_basis_id.size() != triangle_count) ||
      mesh.minus_domain_id.size() != triangle_count ||
      mesh.plus_domain_id.size() != triangle_count ||
      mesh.channel_id.size() != triangle_count ||
      (!mesh.triangle_transport.empty() &&
       mesh.triangle_transport.size() != triangle_count) ||
      (!mesh.triangle_source_quadrature.empty() &&
       mesh.triangle_source_quadrature.size() != triangle_count) ||
      (!mesh.triangle_source_analytic_primitive.empty() &&
       mesh.triangle_source_analytic_primitive.size() != triangle_count))
    throw std::runtime_error("geometry arrays do not share triangle length");
  H5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose);
  create_group(file, "/geometry");
  std::vector<double> vertices;
  vertices.reserve(3 * mesh.vertices.size());
  for (const auto& vertex : mesh.vertices) {
    vertices.push_back(vertex.x);
    vertices.push_back(vertex.y);
    vertices.push_back(vertex.z);
  }
  std::vector<std::uint32_t> triangles;
  triangles.reserve(3 * triangle_count);
  for (const auto& triangle : mesh.triangles)
    triangles.insert(triangles.end(), triangle.begin(), triangle.end());
  write_dataset(file, "/geometry/vertices", vertices,
                {mesh.vertices.size(), 3});
  write_dataset(file, "/geometry/triangles", triangles, {triangle_count, 3});
  write_dataset(file, "/geometry/surface_id", mesh.surface_id,
                {triangle_count});
  std::vector<std::uint32_t> surface_basis_id = mesh.surface_basis_id;
  if (surface_basis_id.empty()) {
    surface_basis_id.resize(triangle_count);
    std::iota(surface_basis_id.begin(), surface_basis_id.end(), 0u);
  }
  write_dataset(file, "/geometry/surface_basis_id", surface_basis_id,
                {triangle_count});
  write_dataset(file, "/geometry/minus_domain_id", mesh.minus_domain_id,
                {triangle_count});
  write_dataset(file, "/geometry/plus_domain_id", mesh.plus_domain_id,
                {triangle_count});
  write_dataset(file, "/geometry/channel_id", mesh.channel_id,
                {triangle_count});
  const std::vector<std::uint8_t> triangle_transport =
      mesh.triangle_transport.empty()
          ? std::vector<std::uint8_t>(triangle_count, 1u)
          : mesh.triangle_transport;
  write_dataset(file, "/geometry/triangle_transport", triangle_transport,
                {triangle_count});
  const std::vector<std::uint8_t> triangle_source_quadrature =
      mesh.triangle_source_quadrature.empty()
          ? triangle_transport
          : mesh.triangle_source_quadrature;
  write_dataset(file, "/geometry/triangle_source_quadrature",
                triangle_source_quadrature, {triangle_count});
  const std::vector<std::uint32_t> triangle_source_analytic_primitive =
      mesh.triangle_source_analytic_primitive.empty()
          ? std::vector<std::uint32_t>(
                triangle_count, std::numeric_limits<std::uint32_t>::max())
          : mesh.triangle_source_analytic_primitive;
  write_dataset(file, "/geometry/triangle_source_analytic_primitive",
                triangle_source_analytic_primitive, {triangle_count});
  if (!mesh.analytic_primitives.empty()) {
    create_group(file, "/analytic");
    std::vector<std::uint8_t> kind;
    std::vector<double> center, axis_x, axis_y, axis_z, parameters;
    std::vector<double> normal_sign;
    std::vector<std::uint8_t> source_integral;
    std::vector<std::uint32_t> surface_id, surface_basis_id;
    std::vector<std::int32_t> minus_domain_id, plus_domain_id, channel_id;
    std::vector<std::uint64_t> surface_element, hole_offset{0};
    std::vector<double> hole_center, hole_radius;
    kind.reserve(mesh.analytic_primitives.size());
    for (const auto& primitive : mesh.analytic_primitives) {
      kind.push_back(static_cast<std::uint8_t>(primitive.kind));
      for (const auto* vector :
           {&primitive.center_mm, &primitive.axis_x, &primitive.axis_y,
            &primitive.axis_z}) {
        auto& output = vector == &primitive.center_mm
                           ? center
                           : vector == &primitive.axis_x
                                 ? axis_x
                                 : vector == &primitive.axis_y ? axis_y
                                                               : axis_z;
        output.insert(output.end(), {vector->x, vector->y, vector->z});
      }
      parameters.insert(parameters.end(), primitive.parameters.begin(),
                        primitive.parameters.end());
      normal_sign.push_back(primitive.normal_sign);
      source_integral.push_back(
          static_cast<std::uint8_t>(primitive.source_integral));
      surface_id.push_back(primitive.surface_id);
      surface_basis_id.push_back(primitive.surface_basis_id);
      minus_domain_id.push_back(primitive.minus_domain_id);
      plus_domain_id.push_back(primitive.plus_domain_id);
      channel_id.push_back(primitive.channel_id);
      surface_element.push_back(primitive.surface_element);
      for (const auto& hole : primitive.holes) {
        hole_center.insert(
            hole_center.end(),
            {hole.center_uv_mm.x, hole.center_uv_mm.y});
        hole_radius.push_back(hole.radius_mm);
      }
      hole_offset.push_back(hole_radius.size());
    }
    const auto count = mesh.analytic_primitives.size();
    write_dataset(file, "/analytic/kind", kind, {count});
    write_dataset(file, "/analytic/center_mm", center, {count, 3});
    write_dataset(file, "/analytic/axis_x", axis_x, {count, 3});
    write_dataset(file, "/analytic/axis_y", axis_y, {count, 3});
    write_dataset(file, "/analytic/axis_z", axis_z, {count, 3});
    write_dataset(file, "/analytic/parameters", parameters, {count, 4});
    write_dataset(file, "/analytic/normal_sign", normal_sign, {count});
    write_dataset(file, "/analytic/source_integral", source_integral,
                  {count});
    write_dataset(file, "/analytic/surface_id", surface_id, {count});
    write_dataset(file, "/analytic/surface_basis_id", surface_basis_id,
                  {count});
    write_dataset(file, "/analytic/minus_domain_id", minus_domain_id,
                  {count});
    write_dataset(file, "/analytic/plus_domain_id", plus_domain_id, {count});
    write_dataset(file, "/analytic/channel_id", channel_id, {count});
    write_dataset(file, "/analytic/surface_element", surface_element,
                  {count});
    write_dataset(file, "/analytic/hole_offset", hole_offset, {count + 1});
    write_dataset(file, "/analytic/hole_center_uv_mm", hole_center,
                  {hole_radius.size(), 2});
    write_dataset(file, "/analytic/hole_radius_mm", hole_radius,
                  {hole_radius.size()});
  }
  if (!mesh.analytic_surface_elements.empty()) {
    if (mesh.analytic_primitives.empty())
      throw std::runtime_error(
          "analytic surface elements require analytic primitives");
    create_group(file, "/analytic/elements");
    std::vector<std::uint32_t> primitive_index, surface_basis_id;
    std::vector<std::uint32_t> aperture_primitive, aperture_hole;
    std::vector<std::uint8_t> coordinates, source_quadrature,
        source_visibility;
    std::vector<double> bounds, center, normal, area;
    std::vector<std::uint64_t> surface_element;
    const auto count = mesh.analytic_surface_elements.size();
    primitive_index.reserve(count);
    coordinates.reserve(count);
    surface_basis_id.reserve(count);
    surface_element.reserve(count);
    for (const auto& element : mesh.analytic_surface_elements) {
      primitive_index.push_back(element.primitive_index);
      coordinates.push_back(static_cast<std::uint8_t>(element.coordinates));
      bounds.insert(bounds.end(), element.bounds.begin(),
                    element.bounds.end());
      center.insert(center.end(),
                    {element.center_mm.x, element.center_mm.y,
                     element.center_mm.z});
      normal.insert(normal.end(),
                    {element.normal.x, element.normal.y, element.normal.z});
      area.push_back(element.area_mm2);
      surface_basis_id.push_back(element.surface_basis_id);
      surface_element.push_back(element.surface_element);
      source_quadrature.push_back(element.source_quadrature ? 1u : 0u);
      source_visibility.push_back(
          static_cast<std::uint8_t>(element.source_visibility));
      aperture_primitive.push_back(
          element.projected_aperture_primitive_index);
      aperture_hole.push_back(element.projected_aperture_hole_index);
    }
    write_dataset(file, "/analytic/elements/primitive_index",
                  primitive_index, {count});
    write_dataset(file, "/analytic/elements/coordinates", coordinates,
                  {count});
    write_dataset(file, "/analytic/elements/bounds", bounds, {count, 5});
    write_dataset(file, "/analytic/elements/center_mm", center, {count, 3});
    write_dataset(file, "/analytic/elements/normal", normal, {count, 3});
    write_dataset(file, "/analytic/elements/area_mm2", area, {count});
    write_dataset(file, "/analytic/elements/surface_basis_id",
                  surface_basis_id, {count});
    write_dataset(file, "/analytic/elements/surface_element",
                  surface_element, {count});
    write_dataset(file, "/analytic/elements/source_quadrature",
                  source_quadrature, {count});
    write_dataset(file, "/analytic/elements/source_visibility",
                  source_visibility, {count});
    write_dataset(
        file,
        "/analytic/elements/projected_aperture_primitive_index",
        aperture_primitive, {count});
    write_dataset(file,
                  "/analytic/elements/projected_aperture_hole_index",
                  aperture_hole, {count});
  }
}

OperatorSet load_operators_hdf5(const std::filesystem::path& path) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  OperatorSet result;
  result.transition = read_csr(file, "/operators/transition");
  result.detection = read_csr(file, "/operators/detection");
  result.losses = read_csr(file, "/operators/losses");
  if (exists(file, "/operators/function_blocks/count")) {
    const auto count =
        read_dataset<std::uint64_t>(file,
                                    "/operators/function_blocks/count")
            .first.at(0);
    result.function_blocks.reserve(count);
    for (std::uint64_t index = 0; index < count; ++index) {
      const std::string root =
          "/operators/function_blocks/" + std::to_string(index);
      const auto descriptor =
          read_dataset<std::uint64_t>(file,
                                      (root + "/descriptor").c_str())
              .first;
      const auto contraction =
          read_dataset<double>(file,
                               (root + "/contraction_bound").c_str())
              .first.at(0);
      if (descriptor.size() != 4)
        throw std::runtime_error(
            "functional block descriptor must contain four integers");
      FunctionBlock block;
      block.name = read_string(file, root + "/name");
      block.library_path = read_string(file, root + "/library_path");
      block.config_json = read_string(file, root + "/config_json");
      block.state_offset = descriptor[0];
      block.state_count = descriptor[1];
      block.egress_count = descriptor[2];
      block.intrinsic_loss_count = descriptor[3];
      block.contraction_bound = contraction;
      block.egress_to_transition =
          read_csr(file, root + "/egress_to_transition");
      block.egress_to_detection =
          read_csr(file, root + "/egress_to_detection");
      block.egress_to_losses =
          read_csr(file, root + "/egress_to_losses");
      block.intrinsic_loss_columns =
          read_dataset<std::uint32_t>(
              file, (root + "/intrinsic_loss_columns").c_str())
              .first;
      result.function_blocks.push_back(std::move(block));
    }
  }
  result.channel_ids =
      read_dataset<std::int32_t>(file, "/operators/channel_id").first;
  if (exists(file, "/metadata/state_labels_json"))
    result.state_labels =
        nlohmann::json::parse(read_string(file, "/metadata/state_labels_json"))
            .get<std::vector<std::string>>();
  if (exists(file, "/metadata/loss_names_json"))
    result.loss_names =
        nlohmann::json::parse(read_string(file, "/metadata/loss_names_json"))
            .get<std::vector<std::string>>();
  else
    for (std::uint64_t i = 0; i < result.losses.cols; ++i)
      result.loss_names.push_back("loss_" + std::to_string(i));
  if (exists(file, "/metadata/energy_eV"))
    result.energy_eV =
        read_dataset<double>(file, "/metadata/energy_eV").first.at(0);
  if (exists(file, "/metadata/ray_origin_offset_mm"))
    result.ray_origin_offset_mm =
        read_dataset<double>(file, "/metadata/ray_origin_offset_mm")
            .first.at(0);
  result.cache_key_sha256 =
      read_string(file, "/metadata/cache_key_sha256");
  result.scene_sha256 = read_string(file, "/metadata/scene_sha256");
  result.geometry_sha256 = read_string(file, "/metadata/geometry_sha256");
  result.surface_basis_sha256 =
      read_string(file, "/metadata/surface_basis_sha256");
  result.dependency_lock_sha256 =
      read_string(file, "/metadata/dependency_lock_sha256");
  result.code_commit = read_string(file, "/metadata/code_commit");
  result.validate();
  return result;
}

void save_operators_hdf5(const std::filesystem::path& path,
                         const OperatorSet& operators) {
  operators.validate();
  H5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose);
  create_group(file, "/operators");
  write_csr(file, "/operators/transition", operators.transition);
  write_csr(file, "/operators/detection", operators.detection);
  write_csr(file, "/operators/losses", operators.losses);
  create_group(file, "/operators/function_blocks");
  write_dataset(
      file, "/operators/function_blocks/count",
      std::vector<std::uint64_t>{operators.function_blocks.size()}, {1});
  for (std::size_t index = 0; index < operators.function_blocks.size();
       ++index) {
    const auto& block = operators.function_blocks[index];
    const std::string root =
        "/operators/function_blocks/" + std::to_string(index);
    create_group(file, root.c_str());
    write_string(file, root + "/name", block.name);
    write_string(file, root + "/library_path", block.library_path);
    write_string(file, root + "/config_json", block.config_json);
    write_dataset(file, (root + "/descriptor").c_str(),
                  std::vector<std::uint64_t>{
                      block.state_offset, block.state_count,
                      block.egress_count, block.intrinsic_loss_count},
                  {4});
    write_dataset(file, (root + "/contraction_bound").c_str(),
                  std::vector<double>{block.contraction_bound}, {1});
    write_csr(file, root + "/egress_to_transition",
              block.egress_to_transition);
    write_csr(file, root + "/egress_to_detection",
              block.egress_to_detection);
    write_csr(file, root + "/egress_to_losses", block.egress_to_losses);
    write_dataset(file, (root + "/intrinsic_loss_columns").c_str(),
                  block.intrinsic_loss_columns,
                  {block.intrinsic_loss_columns.size()});
  }
  write_dataset(file, "/operators/channel_id", operators.channel_ids,
                {operators.channel_ids.size()});
  create_group(file, "/metadata");
  write_dataset(file, "/metadata/energy_eV",
                std::vector<double>{operators.energy_eV}, {1});
  write_dataset(file, "/metadata/ray_origin_offset_mm",
                std::vector<double>{operators.ray_origin_offset_mm}, {1});
  write_string(file, "/metadata/cache_key_sha256",
               operators.cache_key_sha256);
  write_string(file, "/metadata/scene_sha256", operators.scene_sha256);
  write_string(file, "/metadata/geometry_sha256", operators.geometry_sha256);
  write_string(file, "/metadata/surface_basis_sha256",
               operators.surface_basis_sha256);
  write_string(file, "/metadata/dependency_lock_sha256",
               operators.dependency_lock_sha256);
  write_string(file, "/metadata/code_commit", operators.code_commit);
  write_string(file, "/metadata/loss_names_json",
               nlohmann::json(operators.loss_names).dump());
  write_string(file, "/metadata/state_labels_json",
               nlohmann::json(operators.state_labels).dump());
}

SourceBatch load_source_batch_hdf5(const std::filesystem::path& path,
                                   std::uint64_t states,
                                   std::uint64_t channels,
                                   std::uint64_t losses) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  SourceBatch result;
  auto [initial, initial_dims] =
      read_dataset<double>(file, "/sources/initial_states");
  auto [detected, detected_dims] =
      read_dataset<double>(file, "/sources/direct_detection");
  auto [lost, lost_dims] =
      read_dataset<double>(file, "/sources/direct_losses");
  if (initial_dims.size() != 2 || initial_dims[1] != states ||
      detected_dims.size() != 2 || detected_dims[1] != channels ||
      lost_dims.size() != 2 || lost_dims[1] != losses ||
      detected_dims[0] != initial_dims[0] ||
      lost_dims[0] != initial_dims[0]) {
    throw std::runtime_error("source batch datasets have incompatible shapes");
  }
  result.count = initial_dims[0];
  result.initial_states = std::move(initial);
  result.direct_detection = std::move(detected);
  result.direct_losses = std::move(lost);
  if (exists(file, "/sources/source_integration_l1_error_estimate")) {
    auto [estimate, estimate_dims] = read_dataset<double>(
        file, "/sources/source_integration_l1_error_estimate");
    if (estimate_dims != std::vector<hsize_t>{result.count})
      throw std::runtime_error(
          "source integration error-estimate dataset has incompatible shape");
    result.source_integration_l1_error_estimate = std::move(estimate);
  } else {
    result.source_integration_l1_error_estimate.assign(result.count, 0.0);
  }
  return result;
}

void save_response_hdf5(const std::filesystem::path& path,
                        const SolveResult& result,
                        const OperatorSet& operators,
                        std::uint64_t source_count) {
  H5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose);
  H5Handle group(H5Gcreate2(file, "/response", H5P_DEFAULT, H5P_DEFAULT,
                            H5P_DEFAULT),
                 H5Gclose);
  write_dataset(file, "/response/efficiency", result.efficiency,
                {source_count, operators.detection.cols});
  write_dataset(file, "/response/losses", result.losses,
                {source_count, operators.losses.cols});
  write_dataset(file, "/response/unresolved", result.unresolved,
                {source_count});
  write_dataset(file, "/response/input_weight", result.input_weight,
                {source_count});
  if (result.source_integration_l1_error_estimate.size() != source_count)
    throw std::runtime_error(
        "source integration error-estimate shape mismatch");
  write_dataset(file, "/response/source_integration_l1_error_estimate",
                result.source_integration_l1_error_estimate, {source_count});
  if (result.float32_efficiency_loss_upper_bound.size() != source_count)
    throw std::runtime_error(
        "float32 efficiency-loss upper-bound shape mismatch");
  write_dataset(file, "/response/float32_efficiency_loss_upper_bound",
                result.float32_efficiency_loss_upper_bound, {source_count});
  std::vector<double> closure(source_count);
  for (std::uint64_t source = 0; source < source_count; ++source) {
    double terminal = result.unresolved[source];
    for (std::uint64_t channel = 0; channel < operators.detection.cols;
         ++channel)
      terminal +=
          result.efficiency[source * operators.detection.cols + channel];
    for (std::uint64_t loss = 0; loss < operators.losses.cols; ++loss)
      terminal += result.losses[source * operators.losses.cols + loss];
    closure[source] = result.input_weight[source] - terminal;
  }
  write_dataset(file, "/response/energy_balance_error", closure,
                {source_count});
  write_dataset(file, "/response/channel_id", operators.channel_ids,
                {operators.channel_ids.size()});
  write_string(file, "/response/loss_names_json",
               nlohmann::json(operators.loss_names).dump());
  write_dataset(file, "/response/iterations",
                std::vector<std::uint32_t>{result.iterations}, {1});
  write_string(file, "/response/backend", result.backend);
  write_string(file, "/response/hardware", result.hardware);
  write_dataset(file, "/response/wall_seconds",
                std::vector<double>{result.wall_seconds}, {1});
  write_dataset(file, "/response/peak_device_bytes",
                std::vector<std::uint64_t>{result.peak_device_bytes}, {1});
}

void save_effective_response_hdf5(const std::filesystem::path& path,
                                  const EffectiveResponse& response) {
  response.validate();
  H5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose);
  create_group(file, "/effective");
  write_dataset(file, "/effective/state_to_detection",
                response.state_to_detection,
                {response.states, response.channels});
  write_dataset(file, "/effective/state_to_losses", response.state_to_losses,
                {response.states, response.losses});
  write_dataset(file, "/effective/state_unresolved",
                response.state_unresolved, {response.states});
  write_dataset(file, "/effective/channel_id", response.channel_ids,
                {response.channels});
  create_group(file, "/metadata");
  write_string(file, "/metadata/schema",
               "oos.effective-adjoint-response.v3");
  write_string(file, "/metadata/construction_method",
               response.construction_method);
  write_dataset(file, "/metadata/cycles",
                std::vector<std::uint32_t>{response.cycles}, {1});
  write_dataset(file, "/metadata/build_batch_size",
                std::vector<std::uint64_t>{response.build_batch_size}, {1});
  write_dataset(file, "/metadata/operator_tolerance",
                std::vector<double>{response.operator_tolerance}, {1});
  write_string(file, "/metadata/operator_cache_key_sha256",
               response.operator_cache_key_sha256);
  write_string(file, "/metadata/fingerprint_sha256",
               effective_response_fingerprint(response));
  write_string(file, "/metadata/code_commit", response.code_commit);
  write_string(file, "/metadata/loss_names_json",
               nlohmann::json(response.loss_names).dump());
}

EffectiveResponse load_effective_response_hdf5(
    const std::filesystem::path& path) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (read_string(file, "/metadata/schema") !=
      "oos.effective-adjoint-response.v3")
    throw std::runtime_error("unsupported effective response schema");
  EffectiveResponse response;
  auto [detection, detection_dims] =
      read_dataset<double>(file, "/effective/state_to_detection");
  auto [losses, loss_dims] =
      read_dataset<double>(file, "/effective/state_to_losses");
  auto [unresolved, unresolved_dims] =
      read_dataset<double>(file, "/effective/state_unresolved");
  if (detection_dims.size() != 2 || loss_dims.size() != 2 ||
      unresolved_dims.size() != 1 ||
      loss_dims[0] != detection_dims[0] ||
      unresolved_dims[0] != detection_dims[0])
    throw std::runtime_error("effective response datasets have invalid shapes");
  response.states = detection_dims[0];
  response.channels = detection_dims[1];
  response.losses = loss_dims[1];
  response.state_to_detection = std::move(detection);
  response.state_to_losses = std::move(losses);
  response.state_unresolved = std::move(unresolved);
  response.channel_ids =
      read_dataset<std::int32_t>(file, "/effective/channel_id").first;
  response.cycles =
      read_dataset<std::uint32_t>(file, "/metadata/cycles").first.at(0);
  response.build_batch_size =
      read_dataset<std::uint64_t>(file, "/metadata/build_batch_size")
          .first.at(0);
  response.operator_tolerance =
      read_dataset<double>(file, "/metadata/operator_tolerance").first.at(0);
  response.construction_method =
      read_string(file, "/metadata/construction_method");
  response.operator_cache_key_sha256 =
      read_string(file, "/metadata/operator_cache_key_sha256");
  response.fingerprint_sha256 =
      read_string(file, "/metadata/fingerprint_sha256");
  response.code_commit = read_string(file, "/metadata/code_commit");
  response.loss_names =
      nlohmann::json::parse(read_string(file, "/metadata/loss_names_json"))
          .get<std::vector<std::string>>();
  if (response.fingerprint_sha256 !=
      effective_response_fingerprint(response))
    throw std::runtime_error(
        "effective response semantic fingerprint does not match metadata");
  response.validate();
  return response;
}

HitBatch load_hit_batch_hdf5(const std::filesystem::path& path) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  HitBatch result;
  auto [counts, count_dims] =
      read_dataset<std::uint64_t>(file, "/hits/counts");
  if (count_dims.size() != 2 || count_dims[0] == 0 || count_dims[1] == 0)
    throw std::runtime_error("/hits/counts must have shape [events,channels]");
  result.count = count_dims[0];
  result.channels = count_dims[1];
  result.counts = std::move(counts);
  auto [channel, channel_dims] =
      read_dataset<std::int32_t>(file, "/hits/channel_id");
  if (channel_dims != std::vector<hsize_t>{result.channels})
    throw std::runtime_error("/hits/channel_id has incompatible shape");
  result.channel_ids = std::move(channel);
  if (exists(file, "/hits/truth_xy_mm")) {
    auto [truth, truth_dims] =
        read_dataset<double>(file, "/hits/truth_xy_mm");
    if (truth_dims != std::vector<hsize_t>{result.count, 2})
      throw std::runtime_error("/hits/truth_xy_mm has incompatible shape");
    result.truth_xy_mm = std::move(truth);
  }
  return result;
}

void save_response_grid_hdf5(const std::filesystem::path& path,
                             const ResponseGrid& grid) {
  grid.validate();
  if (grid.effective_response_fingerprint_sha256.empty() ||
      grid.source_angular_mode.empty())
    throw std::runtime_error(
        "response grid lacks required source provenance");
  const auto fingerprint = response_grid_fingerprint(grid);
  if (!grid.fingerprint_sha256.empty() &&
      grid.fingerprint_sha256 != fingerprint)
    throw std::runtime_error(
        "response-grid semantic fingerprint is stale");
  H5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose);
  create_group(file, "/grid");
  write_dataset(file, "/grid/xy_mm", grid.xy_mm, {grid.points, 2});
  write_dataset(file, "/grid/conditional_log_probability",
                grid.conditional_log_probability,
                {grid.points, grid.channels});
  write_dataset(file, "/grid/top_efficiency", grid.top_efficiency,
                {grid.points});
  write_dataset(file, "/grid/channel_id", grid.channel_ids,
                {grid.channels});
  create_group(file, "/metadata");
  write_string(file, "/metadata/schema", "oos.response-grid.v2");
  write_string(file, "/metadata/fingerprint_sha256",
               fingerprint);
  write_string(file, "/metadata/domain_shape", grid.domain_shape);
  write_dataset(file, "/metadata/radius_mm",
                std::vector<double>{grid.radius_mm}, {1});
  write_dataset(file, "/metadata/half_x_mm",
                std::vector<double>{grid.half_x_mm}, {1});
  write_dataset(file, "/metadata/half_y_mm",
                std::vector<double>{grid.half_y_mm}, {1});
  write_dataset(file, "/metadata/line_y_start_mm",
                std::vector<double>{grid.line_y_start_mm}, {1});
  write_dataset(file, "/metadata/line_pitch_mm",
                std::vector<double>{grid.line_pitch_mm}, {1});
  write_dataset(file, "/metadata/line_count",
                std::vector<std::uint64_t>{grid.line_count}, {1});
  write_dataset(file, "/metadata/spacing_mm",
                std::vector<double>{grid.spacing_mm}, {1});
  if (!grid.effective_response_sha256.empty())
    write_string(file, "/metadata/effective_response_sha256",
                 grid.effective_response_sha256);
  write_string(file, "/metadata/effective_response_fingerprint_sha256",
               grid.effective_response_fingerprint_sha256);
  write_string(file, "/metadata/source_angular_mode",
               grid.source_angular_mode);
  write_string(file, "/metadata/source_backend", grid.source_backend);
  write_dataset(file, "/metadata/source_z_mm",
                std::vector<double>{grid.source_z_mm}, {1});
  write_dataset(file, "/metadata/source_thickness_mm",
                std::vector<double>{grid.source_thickness_mm}, {1});
  write_dataset(file, "/metadata/source_transverse_count",
                std::vector<std::uint32_t>{grid.source_transverse_count},
                {1});
  write_dataset(file, "/metadata/obstacle_half_width_mm",
                std::vector<double>{grid.obstacle_half_width_mm}, {1});
  write_dataset(file, "/metadata/obstacle_half_thickness_mm",
                std::vector<double>{grid.obstacle_half_thickness_mm}, {1});
  write_dataset(file, "/metadata/source_medium_z_max_mm",
                std::vector<double>{grid.source_medium_z_max_mm}, {1});
  write_dataset(file, "/metadata/source_mu_order",
                std::vector<std::uint32_t>{grid.source_mu_order}, {1});
  write_dataset(file, "/metadata/source_phi_count",
                std::vector<std::uint32_t>{grid.source_phi_count}, {1});
  write_dataset(file, "/metadata/source_relative_tolerance",
                std::vector<double>{grid.source_relative_tolerance}, {1});
  write_dataset(
      file, "/metadata/source_maximum_subdivision_depth",
      std::vector<std::uint32_t>{grid.source_maximum_subdivision_depth}, {1});
  write_dataset(file, "/metadata/structured_disk_mu_order",
                std::vector<std::uint32_t>{grid.structured_disk_mu_order},
                {1});
  write_dataset(file, "/metadata/structured_disk_phi_count",
                std::vector<std::uint32_t>{grid.structured_disk_phi_count},
                {1});
}

ResponseGrid load_response_grid_hdf5(const std::filesystem::path& path) {
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (read_string(file, "/metadata/schema") != "oos.response-grid.v2")
    throw std::runtime_error("unsupported response-grid schema");
  ResponseGrid grid;
  grid.fingerprint_sha256 =
      read_string(file, "/metadata/fingerprint_sha256");
  auto [xy, xy_dims] = read_dataset<double>(file, "/grid/xy_mm");
  auto [log_probability, probability_dims] =
      read_dataset<float>(file, "/grid/conditional_log_probability");
  auto [efficiency, efficiency_dims] =
      read_dataset<double>(file, "/grid/top_efficiency");
  if (xy_dims.size() != 2 || xy_dims[1] != 2 ||
      probability_dims.size() != 2 ||
      probability_dims[0] != xy_dims[0] ||
      efficiency_dims != std::vector<hsize_t>{xy_dims[0]})
    throw std::runtime_error("response-grid datasets have invalid shapes");
  grid.points = xy_dims[0];
  grid.channels = probability_dims[1];
  grid.xy_mm = std::move(xy);
  grid.conditional_log_probability = std::move(log_probability);
  grid.top_efficiency = std::move(efficiency);
  grid.channel_ids =
      read_dataset<std::int32_t>(file, "/grid/channel_id").first;
  grid.domain_shape = exists(file, "/metadata/domain_shape")
                          ? read_string(file, "/metadata/domain_shape")
                          : "disk";
  grid.radius_mm =
      read_dataset<double>(file, "/metadata/radius_mm").first.at(0);
  if (exists(file, "/metadata/half_x_mm"))
    grid.half_x_mm =
        read_dataset<double>(file, "/metadata/half_x_mm").first.at(0);
  if (exists(file, "/metadata/half_y_mm"))
    grid.half_y_mm =
        read_dataset<double>(file, "/metadata/half_y_mm").first.at(0);
  if (exists(file, "/metadata/line_y_start_mm"))
    grid.line_y_start_mm =
        read_dataset<double>(file, "/metadata/line_y_start_mm").first.at(0);
  if (exists(file, "/metadata/line_pitch_mm"))
    grid.line_pitch_mm =
        read_dataset<double>(file, "/metadata/line_pitch_mm").first.at(0);
  if (exists(file, "/metadata/line_count"))
    grid.line_count =
        read_dataset<std::uint64_t>(file, "/metadata/line_count").first.at(0);
  grid.spacing_mm =
      read_dataset<double>(file, "/metadata/spacing_mm").first.at(0);
  if (exists(file, "/metadata/effective_response_sha256"))
    grid.effective_response_sha256 =
        read_string(file, "/metadata/effective_response_sha256");
  grid.effective_response_fingerprint_sha256 = read_string(
      file, "/metadata/effective_response_fingerprint_sha256");
  grid.source_angular_mode =
      read_string(file, "/metadata/source_angular_mode");
  grid.source_backend = read_string(file, "/metadata/source_backend");
  grid.source_z_mm =
      read_dataset<double>(file, "/metadata/source_z_mm").first.at(0);
  grid.source_thickness_mm =
      read_dataset<double>(file, "/metadata/source_thickness_mm")
          .first.at(0);
  grid.source_transverse_count =
      read_dataset<std::uint32_t>(
          file, "/metadata/source_transverse_count").first.at(0);
  grid.obstacle_half_width_mm =
      read_dataset<double>(file, "/metadata/obstacle_half_width_mm")
          .first.at(0);
  grid.obstacle_half_thickness_mm =
      read_dataset<double>(file, "/metadata/obstacle_half_thickness_mm")
          .first.at(0);
  grid.source_medium_z_max_mm =
      read_dataset<double>(file, "/metadata/source_medium_z_max_mm")
          .first.at(0);
  grid.source_mu_order =
      read_dataset<std::uint32_t>(
          file, "/metadata/source_mu_order").first.at(0);
  grid.source_phi_count =
      read_dataset<std::uint32_t>(
          file, "/metadata/source_phi_count").first.at(0);
  grid.source_relative_tolerance =
      read_dataset<double>(
          file, "/metadata/source_relative_tolerance").first.at(0);
  grid.source_maximum_subdivision_depth =
      read_dataset<std::uint32_t>(
          file, "/metadata/source_maximum_subdivision_depth").first.at(0);
  grid.structured_disk_mu_order =
      read_dataset<std::uint32_t>(
          file, "/metadata/structured_disk_mu_order").first.at(0);
  grid.structured_disk_phi_count =
      read_dataset<std::uint32_t>(
          file, "/metadata/structured_disk_phi_count").first.at(0);
  grid.validate();
  if (grid.fingerprint_sha256 != response_grid_fingerprint(grid))
    throw std::runtime_error(
        "response-grid semantic fingerprint does not match metadata");
  return grid;
}

void save_regression_hdf5(const std::filesystem::path& path,
                          const RegressionResult& result,
                          const std::vector<std::int32_t>& channel_ids) {
  if (result.events == 0 ||
      result.fitted_xy_mm.size() != result.events * 2 ||
      result.log_likelihood.size() != result.events ||
      (!result.fitted_line_id.empty() &&
       result.fitted_line_id.size() != result.events) ||
      (!result.fitted_line_x_mm.empty() &&
       result.fitted_line_x_mm.size() != result.events) ||
      (result.fitted_line_id.empty() !=
       result.fitted_line_x_mm.empty()) ||
      (!result.error_mm.empty() && result.error_mm.size() != result.events))
    throw std::runtime_error("regression result dimensions are invalid");
  H5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose);
  create_group(file, "/regression");
  write_dataset(file, "/regression/fitted_xy_mm", result.fitted_xy_mm,
                {result.events, 2});
  write_dataset(file, "/regression/log_likelihood", result.log_likelihood,
                {result.events});
  if (!result.fitted_line_id.empty()) {
    write_dataset(file, "/regression/fitted_line_id",
                  result.fitted_line_id, {result.events});
    write_dataset(file, "/regression/fitted_line_x_mm",
                  result.fitted_line_x_mm, {result.events});
  }
  if (!result.error_mm.empty())
    write_dataset(file, "/regression/error_mm", result.error_mm,
                  {result.events});
  write_dataset(file, "/regression/channel_id", channel_ids,
                {channel_ids.size()});
  if (!result.full_plane_log_likelihood.empty()) {
    if (result.likelihood_points == 0 ||
        result.full_plane_log_likelihood.size() !=
            result.events * result.likelihood_points)
      throw std::runtime_error("full-plane likelihood dimensions are invalid");
    write_dataset(file, "/regression/full_plane_log_likelihood",
                  result.full_plane_log_likelihood,
                  {result.events, result.likelihood_points});
  }
  create_group(file, "/metadata");
  write_string(file, "/metadata/schema", "oos.regression.v1");
}

}  // namespace oos
