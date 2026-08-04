#include "oos/plugin.h"

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "lxe_function.hpp"

namespace {
std::string message;

template <typename T>
hid_t native_type();
template <>
hid_t native_type<double>() {
  return H5T_NATIVE_DOUBLE;
}
template <>
hid_t native_type<std::uint64_t>() {
  return H5T_NATIVE_UINT64;
}
template <>
hid_t native_type<std::uint32_t>() {
  return H5T_NATIVE_UINT32;
}
template <>
hid_t native_type<std::int32_t>() {
  return H5T_NATIVE_INT32;
}
template <>
hid_t native_type<std::uint8_t>() {
  return H5T_NATIVE_UINT8;
}

template <typename T>
std::pair<std::vector<T>, std::vector<std::uint64_t>> read(
    hid_t file, const std::string& path) {
  const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dataset < 0) throw std::runtime_error("missing " + path);
  const hid_t space = H5Dget_space(dataset);
  const int rank = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> dimensions(rank);
  H5Sget_simple_extent_dims(space, dimensions.data(), nullptr);
  std::size_t count = 1;
  std::vector<std::uint64_t> shape;
  for (const auto dimension : dimensions) {
    count *= dimension;
    shape.push_back(dimension);
  }
  std::vector<T> values(count);
  if (H5Dread(dataset, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
              values.data()) < 0) {
    H5Sclose(space);
    H5Dclose(dataset);
    throw std::runtime_error("cannot read " + path);
  }
  H5Sclose(space);
  H5Dclose(dataset);
  return {std::move(values), std::move(shape)};
}

oos_string_view_v1 view(const std::string& value) {
  return {value.data(), value.size()};
}

std::string read_string(hid_t file, const std::string& path) {
  const auto [bytes, shape] = read<std::uint8_t>(file, path);
  if (shape.size() != 1)
    throw std::runtime_error("string dataset is not one-dimensional");
  return {bytes.begin(), bytes.end()};
}

bool exists(hid_t file, const std::string& path) {
  return H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0;
}

std::string block_path(const nlohmann::json& configuration) {
  if (configuration.contains("factorized_block_hdf5"))
    return configuration.at("factorized_block_hdf5").get<std::string>();
  return configuration.value("precomputed_block_hdf5", std::string{});
}

LXeFunctionInstance load_function_instance(const std::string& path) {
  const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) throw std::runtime_error("cannot open LXe function block");
  try {
    const auto [real, coefficient_shape] =
        read<double>(file, "/function/coefficients_real");
    const auto [imaginary, imaginary_shape] =
        read<double>(file, "/function/coefficients_imag");
    const auto [expected, expected_shape] =
        read<double>(file, "/function/expected_return");
    const auto [ring_area, ring_shape] =
        read<double>(file, "/function/surface_ring_area_mm2");
    const auto [angular, angular_shape] =
        read<double>(file, "/function/angular_weight");
    const auto metadata =
        nlohmann::json::parse(read_string(file, "/metadata/generator_json"));
    const auto phase =
        nlohmann::json::parse(read_string(file, "/metadata/phase_grid_json"));
    if (coefficient_shape.size() != 5 ||
        coefficient_shape != imaginary_shape ||
        real.size() != imaginary.size() ||
        expected_shape !=
            std::vector<std::uint64_t>{coefficient_shape[0],
                                       coefficient_shape[1],
                                       coefficient_shape[2]} ||
        ring_shape !=
            std::vector<std::uint64_t>{coefficient_shape[4]} ||
        angular_shape.size() != 1 ||
        metadata.value("schema", std::string{}) !=
            "oos.nonlocal.function.v1")
      throw std::runtime_error("LXe function block has inconsistent shapes");
    LXeFunctionInstance result;
    result.nd = coefficient_shape[0];
    result.nr = coefficient_shape[1];
    result.nm = coefficient_shape[2];
    result.orders = coefficient_shape[3];
    result.surface_radial = coefficient_shape[4];
    result.np = phase.at("position_phi_bins").get<std::uint64_t>();
    result.surface_phi =
        metadata.at("surface_phi_bins").get<std::uint64_t>();
    result.angular = angular.size();
    if (phase.at("position_radial_bins").get<std::uint64_t>() != result.nr ||
        phase.at("direction_mu_bins").get<std::uint64_t>() != result.nm ||
        phase.at("direction_phi_bins").get<std::uint64_t>() != result.nd ||
        metadata.at("angular_count").get<std::uint64_t>() != result.angular)
      throw std::runtime_error(
          "LXe phase metadata does not match factorized arrays");
    result.coefficients.resize(real.size());
    for (std::size_t index = 0; index < real.size(); ++index)
      result.coefficients[index] = {real[index], imaginary[index]};
    result.expected_return = expected;
    result.surface_ring_area = ring_area;
    result.angular_weight = angular;
    H5Fclose(file);
    return result;
  } catch (...) {
    H5Fclose(file);
    throw;
  }
}

std::pair<std::uint64_t, std::uint64_t> bounded_stencil(
    double value, const std::vector<double>& nodes, double& fraction) {
  if (value <= nodes.front()) {
    fraction = 0.0;
    return {0, 0};
  }
  if (value >= nodes.back()) {
    fraction = 0.0;
    return {nodes.size() - 1, nodes.size() - 1};
  }
  const auto upper = static_cast<std::uint64_t>(
      std::upper_bound(nodes.begin(), nodes.end(), value) - nodes.begin());
  const auto lower = upper - 1;
  fraction = (value - nodes[lower]) / (nodes[upper] - nodes[lower]);
  return {lower, upper};
}

std::pair<std::uint64_t, std::uint64_t> periodic_stencil(
    double angle, std::uint64_t count, double& fraction) {
  if (count == 1) {
    fraction = 0.0;
    return {0, 0};
  }
  const double two_pi = 6.283185307179586476925286766559;
  double wrapped = std::fmod(angle, two_pi);
  if (wrapped < 0.0) wrapped += two_pi;
  const double scaled = wrapped * count / two_pi;
  const auto lower_unwrapped =
      static_cast<std::uint64_t>(std::floor(scaled));
  fraction = scaled - std::floor(scaled);
  return {lower_unwrapped % count, (lower_unwrapped + 1) % count};
}

oos_plugin_validation_v1 validate(oos_string_view_v1 config,
                                  double energy_eV) {
  try {
    if (!(energy_eV > 0.0)) throw std::runtime_error("energy must be positive");
    const std::string value(config.data ? config.data : "", config.size);
    const auto parsed = nlohmann::json::parse(value.empty() ? "{}" : value);
    if (parsed.value("geometry", "finite_cylinder") != "finite_cylinder")
      throw std::runtime_error("LXe v1 supports only finite_cylinder geometry");
    if (parsed.value("explicit_collision_order", 7) != 7)
      throw std::runtime_error("LXe v1 requires explicit_collision_order=7");
    if (parsed.value("replaces_full_operator", false))
      throw std::runtime_error(
          "full-operator replacement is forbidden; provide a nonlocal block");
    const auto path = block_path(parsed);
    if (path.empty())
      throw std::runtime_error(
          "factorized_block_hdf5 or precomputed_block_hdf5 is required");
    if (!std::filesystem::is_regular_file(path))
      throw std::runtime_error("precomputed LXe operator does not exist");
    message.clear();
    return {0, {nullptr, 0}};
  } catch (const std::exception& exception) {
    message = exception.what();
    return {1, {message.data(), message.size()}};
  }
}

int32_t build(oos_string_view_v1 config, double,
              const oos_operator_sink_v1* sink) {
  try {
    const std::string value(config.data ? config.data : "", config.size);
    const auto parsed = nlohmann::json::parse(value);
    const auto path = block_path(parsed);
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("cannot open LXe operator HDF5");
    const bool functional = exists(file, "/function/coefficients_real");
    if (functional) {
      const std::vector<std::string> f64_paths{
          "/nonlocal/egress/barycentric",
          "/nonlocal/egress/direction_local",
          "/nonlocal/egress/stokes",
          "/nonlocal/egress/reference_axis_local"};
      const std::vector<std::string> u64_paths{
          "/nonlocal/egress/surface_element",
          "/nonlocal/egress/side"};
      for (const auto& target_path : f64_paths) {
        const auto [values, shape] = read<double>(file, target_path);
        if (sink->write_dataset_f64(sink->user_data, view(target_path),
                                    values.data(), shape.data(),
                                    shape.size()) != 0)
          throw std::runtime_error("operator sink rejected " + target_path);
      }
      for (const auto& target_path : u64_paths) {
        const auto [values, shape] = read<std::uint64_t>(file, target_path);
        if (sink->write_dataset_u64(sink->user_data, view(target_path),
                                    values.data(), shape.data(),
                                    shape.size()) != 0)
          throw std::runtime_error("operator sink rejected " + target_path);
      }
      const auto loss_names = nlohmann::json::parse(
          read_string(file, "/metadata/loss_names_json"));
      const auto generator = nlohmann::json::parse(
          read_string(file, "/metadata/generator_json"));
      H5Fclose(file);
      const auto metadata =
          nlohmann::json{
              {"mode", "nonlocal_function_block"},
              {"locality", "nonlocal"},
              {"execution", "function"},
              {"state_count", generator.at("state_count")},
              {"explicit_collision_order", 7},
              {"source", path},
              {"loss_names", loss_names}}
              .dump();
      if (sink->write_metadata_json(sink->user_data, view(metadata)) != 0)
        throw std::runtime_error("operator sink rejected metadata");
      return 0;
    }
    if (!exists(file, "/nonlocal/internal_transition/shape") ||
        !exists(file, "/nonlocal/emission/shape") ||
        !exists(file, "/nonlocal/internal_losses/shape"))
      throw std::runtime_error(
          "LXe block must use the geometry-independent nonlocal schema");
    const std::string source_root = "/nonlocal";
    const std::vector<std::pair<std::string, std::string>> f64_paths{
        {source_root + "/internal_transition/data",
         "/nonlocal/internal_transition/data"},
        {source_root + "/emission/data", "/nonlocal/emission/data"},
        {source_root + "/internal_losses/data",
         "/nonlocal/internal_losses/data"},
        {source_root + "/egress/barycentric",
         "/nonlocal/egress/barycentric"},
        {source_root + "/egress/direction_local",
         "/nonlocal/egress/direction_local"}};
    const std::vector<std::pair<std::string, std::string>> u64_paths{
        {source_root + "/internal_transition/shape",
         "/nonlocal/internal_transition/shape"},
        {source_root + "/internal_transition/indptr",
         "/nonlocal/internal_transition/indptr"},
        {source_root + "/internal_transition/indices",
         "/nonlocal/internal_transition/indices"},
        {source_root + "/emission/shape", "/nonlocal/emission/shape"},
        {source_root + "/emission/indptr", "/nonlocal/emission/indptr"},
        {source_root + "/emission/indices", "/nonlocal/emission/indices"},
        {source_root + "/internal_losses/shape",
         "/nonlocal/internal_losses/shape"},
        {source_root + "/internal_losses/indptr",
         "/nonlocal/internal_losses/indptr"},
        {source_root + "/internal_losses/indices",
         "/nonlocal/internal_losses/indices"},
        {source_root + "/egress/surface_element",
         "/nonlocal/egress/surface_element"},
        {source_root + "/egress/side", "/nonlocal/egress/side"}};
    for (const auto& [source_path, target_path] : f64_paths) {
      const auto [values, shape] = read<double>(file, source_path);
      if (sink->write_dataset_f64(sink->user_data, view(target_path),
                                  values.data(), shape.data(),
                                  shape.size()) != 0)
        throw std::runtime_error("operator sink rejected " + target_path);
    }
    for (const auto& [source_path, target_path] : u64_paths) {
      std::vector<std::uint64_t> values;
      std::vector<std::uint64_t> shape;
      if (source_path == source_root + "/internal_transition/indices" ||
          source_path == source_root + "/emission/indices" ||
          source_path == source_root + "/internal_losses/indices") {
        const auto source = read<std::uint32_t>(file, source_path);
        values.assign(source.first.begin(), source.first.end());
        shape = source.second;
      } else {
        auto source = read<std::uint64_t>(file, source_path);
        values = std::move(source.first);
        shape = std::move(source.second);
      }
      if (sink->write_dataset_u64(sink->user_data, view(target_path),
                                  values.data(), shape.data(),
                                  shape.size()) != 0)
        throw std::runtime_error("operator sink rejected " + target_path);
    }
    const auto loss_names = nlohmann::json::parse(
        read_string(file, "/metadata/loss_names_json"));
    H5Fclose(file);
    const auto metadata =
        nlohmann::json{{"mode", "nonlocal_operator_block"},
                       {"locality", "nonlocal"},
                       {"explicit_collision_order", 7},
                       {"source", path},
                       {"loss_names", loss_names}}
            .dump();
    if (sink->write_metadata_json(sink->user_data, view(metadata)) != 0)
      throw std::runtime_error("operator sink rejected metadata");
    return 0;
  } catch (const std::exception& exception) {
    message = exception.what();
    return 2;
  }
}

int32_t deposit(oos_string_view_v1 config, double,
                const oos_surface_hit_v3* hit, std::uint64_t* indices,
                double* weights, std::size_t capacity, std::size_t* count) {
  try {
    const auto parsed = nlohmann::json::parse(
        std::string(config.data ? config.data : "", config.size));
    const auto grid = parsed.value("phase_grid", nlohmann::json::object());
    const std::uint64_t nr = grid.value("position_radial_bins", 1u);
    const std::uint64_t np = grid.value("position_phi_bins", 1u);
    const std::uint64_t nm = grid.value("direction_mu_bins", 1u);
    const std::uint64_t nd = grid.value("direction_phi_bins", 1u);
    if (nr * np * nm * nd == 1) {
      if (capacity < 1) return 3;
      indices[0] = 0;
      weights[0] = 1.0;
      *count = 1;
      return 0;
    }
    const double radius = parsed.at("radius_mm").get<double>();
    const auto center =
        parsed.value("center_xy_mm", std::vector<double>{0.0, 0.0});
    const auto inward =
        parsed.value("inward_normal", std::vector<double>{0.0, 0.0, -1.0});
    const double px = hit->point_local_mm[0] - center.at(0);
    const double py = hit->point_local_mm[1] - center.at(1);
    const double r = std::min(radius, std::hypot(px, py));
    double position_phi = std::atan2(py, px);
    const double mu = std::clamp(
        hit->direction_local[0] * inward.at(0) +
            hit->direction_local[1] * inward.at(1) +
            hit->direction_local[2] * inward.at(2),
        grid.value("direction_mu_minimum", 0.0), 1.0);
    double direction_phi =
        std::atan2(hit->direction_local[1], hit->direction_local[0]) -
        position_phi;
    std::vector<double> radial_nodes(nr);
    for (std::uint64_t i = 0; i < nr; ++i)
      radial_nodes[i] =
          radius * 0.5 *
          (1.0 - std::cos(3.14159265358979323846 * i / nr));
    std::vector<double> mu_nodes(nm);
    const double mu_minimum = grid.value("direction_mu_minimum", 0.0);
    for (std::uint64_t i = 0; i < nm; ++i)
      mu_nodes[i] =
          nm == 1 ? 0.5 : mu_minimum + (1.0 - mu_minimum) * i / (nm - 1);
    double fr = 0.0, fp = 0.0, fm = 0.0, fd = 0.0;
    const auto sr = bounded_stencil(r, radial_nodes, fr);
    const auto sp = periodic_stencil(position_phi, np, fp);
    const auto sm = bounded_stencil(mu, mu_nodes, fm);
    const auto sd = periodic_stencil(direction_phi, nd, fd);
    const std::array<std::pair<std::uint64_t, double>, 2> ar{{
        {sr.first, 1.0 - fr}, {sr.second, fr}}};
    const std::array<std::pair<std::uint64_t, double>, 2> ap{{
        {sp.first, 1.0 - fp}, {sp.second, fp}}};
    const std::array<std::pair<std::uint64_t, double>, 2> am{{
        {sm.first, 1.0 - fm}, {sm.second, fm}}};
    const std::array<std::pair<std::uint64_t, double>, 2> ad{{
        {sd.first, 1.0 - fd}, {sd.second, fd}}};
    std::map<std::uint64_t, double> accumulated;
    for (const auto [ir, wr] : ar)
      for (const auto [ip, wp] : ap)
        for (const auto [im, wm] : am)
          for (const auto [id, wd] : ad) {
            const double weight = wr * wp * wm * wd;
            if (weight == 0.0) continue;
            const auto index = (((ir * np + ip) * nm + im) * nd + id);
            accumulated[index] += weight;
          }
    if (accumulated.size() > capacity) return 3;
    *count = 0;
    for (const auto& [index, weight] : accumulated) {
      indices[*count] = index;
      weights[*count] = weight;
      ++*count;
    }
    return 0;
  } catch (...) {
    return 2;
  }
}

oos_plugin_validation_v1 validate_function(oos_string_view_v1 config,
                                           double energy_eV) {
  return validate(config, energy_eV);
}

int32_t create_function(oos_string_view_v1 config, double,
                        void** instance,
                        oos_function_operator_descriptor_v2* descriptor) {
  try {
    const auto parsed = nlohmann::json::parse(
        std::string(config.data ? config.data : "", config.size));
    auto value =
        std::make_unique<LXeFunctionInstance>(
            load_function_instance(block_path(parsed)));
    const auto states = value->nr * value->np * value->nm * value->nd;
    const auto egress =
        value->surface_radial * value->surface_phi * value->angular;
    const double contraction =
        *std::max_element(value->expected_return.begin(),
                          value->expected_return.end());
    *descriptor = {OOS_FUNCTION_OPERATOR_ABI_V2, states, states, egress, 1,
                   contraction, 1,
#ifdef OOS_LXE_HAS_CUDA
                   1
#else
                   0
#endif
    };
    *instance = value.release();
    return 0;
  } catch (const std::exception& exception) {
    message = exception.what();
    return 1;
  }
}

void destroy_function(void* instance) {
  auto* function = static_cast<LXeFunctionInstance*>(instance);
  destroy_lxe_function_cuda(function);
  delete function;
}

int32_t apply_function_cpu(
    void* opaque, std::uint64_t batch, const double* input,
    double* retained, double* egress, double* losses) {
  try {
    const auto& function = *static_cast<LXeFunctionInstance*>(opaque);
    const auto state_count =
        function.nr * function.np * function.nm * function.nd;
    const auto surface_count =
        function.surface_radial * function.surface_phi;
    const auto egress_count = surface_count * function.angular;
    std::fill(retained, retained + batch * state_count, 0.0);
    std::fill(egress, egress + batch * egress_count, 0.0);
    std::fill(losses, losses + batch, 0.0);
    constexpr double two_pi =
        6.283185307179586476925286766559005768;
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_row = 0;
         signed_row < static_cast<std::int64_t>(batch); ++signed_row) {
      const auto row = static_cast<std::uint64_t>(signed_row);
      std::vector<std::complex<double>> modal(
          function.orders * function.surface_radial, {0.0, 0.0});
      double input_weight = 0.0;
      double expected_weight = 0.0;
      for (std::uint64_t radial = 0; radial < function.nr; ++radial)
        for (std::uint64_t mu = 0; mu < function.nm; ++mu)
          for (std::uint64_t direction_phi = 0;
               direction_phi < function.nd; ++direction_phi) {
            const auto expected_index =
                (direction_phi * function.nr + radial) * function.nm + mu;
            std::vector<std::complex<double>> position_modes(
                function.orders, {0.0, 0.0});
            for (std::uint64_t position_phi = 0;
                 position_phi < function.np; ++position_phi) {
              const auto state =
                  (((radial * function.np + position_phi) * function.nm +
                    mu) *
                       function.nd +
                   direction_phi);
              const double value = input[row * state_count + state];
              input_weight += value;
              expected_weight +=
                  value * function.expected_return[expected_index];
              const double phi =
                  two_pi * static_cast<double>(position_phi) /
                  static_cast<double>(function.np);
              for (std::uint64_t order = 0; order < function.orders;
                   ++order)
                position_modes[order] +=
                    value * std::exp(std::complex<double>{
                                0.0, -static_cast<double>(order) * phi});
            }
            for (std::uint64_t order = 0; order < function.orders; ++order)
              for (std::uint64_t surface = 0;
                   surface < function.surface_radial; ++surface) {
                const auto coefficient =
                    ((((direction_phi * function.nr + radial) *
                           function.nm +
                       mu) *
                          function.orders +
                      order) *
                         function.surface_radial +
                     surface);
                modal[order * function.surface_radial + surface] +=
                    position_modes[order] *
                    function.coefficients[coefficient];
              }
          }
      for (std::uint64_t surface = 0;
           surface < function.surface_radial; ++surface)
        for (std::uint64_t position_phi = 0;
             position_phi < function.surface_phi; ++position_phi) {
          const double phi =
              two_pi * (static_cast<double>(position_phi) + 0.5) /
              static_cast<double>(function.surface_phi);
          std::complex<double> density{0.0, 0.0};
          for (std::uint64_t order = 0; order < function.orders; ++order)
            density +=
                modal[order * function.surface_radial + surface] *
                std::exp(std::complex<double>{
                    0.0, static_cast<double>(order) * phi});
          const double surface_weight =
              density.real() * function.surface_ring_area[surface] /
              static_cast<double>(function.surface_phi);
          const auto surface_index =
              surface * function.surface_phi + position_phi;
          for (std::uint64_t angle = 0; angle < function.angular; ++angle)
            egress[row * egress_count +
                   surface_index * function.angular + angle] =
                surface_weight * function.angular_weight[angle];
        }
      losses[row] = input_weight - expected_weight;
    }
    return 0;
  } catch (const std::exception& exception) {
    message = exception.what();
    return 2;
  }
}

int32_t apply_function_adjoint_cpu(
    void* opaque, std::uint64_t batch, const double* retained_adjoint,
    const double* egress_adjoint, const double* losses_adjoint,
    double* input_adjoint) {
  try {
    const auto& function = *static_cast<LXeFunctionInstance*>(opaque);
    const auto state_count =
        function.nr * function.np * function.nm * function.nd;
    const auto surface_count =
        function.surface_radial * function.surface_phi;
    const auto egress_count = surface_count * function.angular;
    (void)retained_adjoint;
    std::fill(input_adjoint, input_adjoint + batch * state_count, 0.0);
    constexpr double two_pi =
        6.283185307179586476925286766559005768;
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_row = 0;
         signed_row < static_cast<std::int64_t>(batch); ++signed_row) {
      const auto row = static_cast<std::uint64_t>(signed_row);
      std::vector<std::complex<double>> surface_modes(
          function.orders * function.surface_radial, {0.0, 0.0});
      for (std::uint64_t surface = 0;
           surface < function.surface_radial; ++surface)
        for (std::uint64_t position_phi = 0;
             position_phi < function.surface_phi; ++position_phi) {
          const auto surface_index =
              surface * function.surface_phi + position_phi;
          double seed = 0.0;
          for (std::uint64_t angle = 0; angle < function.angular; ++angle)
            seed += egress_adjoint[
                        row * egress_count +
                        surface_index * function.angular + angle] *
                    function.angular_weight[angle];
          seed *= function.surface_ring_area[surface] /
                  static_cast<double>(function.surface_phi);
          const double phi =
              two_pi * (static_cast<double>(position_phi) + 0.5) /
              static_cast<double>(function.surface_phi);
          for (std::uint64_t order = 0; order < function.orders; ++order)
            surface_modes[order * function.surface_radial + surface] +=
                seed * std::exp(std::complex<double>{
                           0.0, static_cast<double>(order) * phi});
        }
      for (std::uint64_t radial = 0; radial < function.nr; ++radial)
        for (std::uint64_t mu = 0; mu < function.nm; ++mu)
          for (std::uint64_t direction_phi = 0;
               direction_phi < function.nd; ++direction_phi) {
            const auto expected_index =
                (direction_phi * function.nr + radial) * function.nm + mu;
            std::vector<std::complex<double>> phase_adjoint(
                function.orders, {0.0, 0.0});
            for (std::uint64_t order = 0; order < function.orders; ++order)
              for (std::uint64_t surface = 0;
                   surface < function.surface_radial; ++surface) {
                const auto coefficient =
                    ((((direction_phi * function.nr + radial) *
                           function.nm +
                       mu) *
                          function.orders +
                      order) *
                         function.surface_radial +
                     surface);
                phase_adjoint[order] +=
                    function.coefficients[coefficient] *
                    surface_modes[order * function.surface_radial + surface];
              }
            for (std::uint64_t position_phi = 0;
                 position_phi < function.np; ++position_phi) {
              const double phi =
                  two_pi * static_cast<double>(position_phi) /
                  static_cast<double>(function.np);
              double value =
                  losses_adjoint[row] *
                  (1.0 - function.expected_return[expected_index]);
              for (std::uint64_t order = 0; order < function.orders;
                   ++order)
                value +=
                    (phase_adjoint[order] *
                     std::exp(std::complex<double>{
                         0.0, -static_cast<double>(order) * phi}))
                        .real();
              const auto state =
                  (((radial * function.np + position_phi) * function.nm +
                    mu) *
                       function.nd +
                   direction_phi);
              input_adjoint[row * state_count + state] = value;
            }
          }
    }
    return 0;
  } catch (const std::exception& exception) {
    message = exception.what();
    return 2;
  }
}

int32_t prepare_function_cuda(void* instance, std::int32_t device) {
  return prepare_lxe_function_cuda(
      static_cast<LXeFunctionInstance*>(instance), device);
}

int32_t apply_function_cuda(
    void* instance, std::uint64_t batch, const double* device_input,
    double* device_retained, double* device_egress, double* device_losses,
    void* cuda_stream) {
  return apply_lxe_function_cuda(
      static_cast<LXeFunctionInstance*>(instance), batch, device_input,
      device_retained, device_egress, device_losses, cuda_stream);
}

int32_t apply_function_adjoint_cuda(
    void* instance, std::uint64_t batch,
    const double* device_retained_adjoint,
    const double* device_egress_adjoint,
    const double* device_losses_adjoint, double* device_input_adjoint,
    void* cuda_stream) {
  return apply_lxe_function_adjoint_cuda(
      static_cast<LXeFunctionInstance*>(instance), batch,
      device_retained_adjoint, device_egress_adjoint,
      device_losses_adjoint, device_input_adjoint, cuda_stream);
}

const oos_surface_plugin_v3 plugin{
    OOS_SURFACE_PLUGIN_ABI_V3,
    OOS_SURFACE_NONLOCAL_V3,
    {"oos_lxe_finite_cylinder", 23},
    {"0.3.0", 5},
    validate,
    build,
    nullptr,
    deposit,
    16,
};

const oos_function_operator_v2 function_operator{
    OOS_FUNCTION_OPERATOR_ABI_V2,
    {"oos_lxe_factorized", 18},
    {"0.5.0", 5},
    validate_function,
    create_function,
    destroy_function,
    apply_function_cpu,
    apply_function_adjoint_cpu,
#ifdef OOS_LXE_HAS_CUDA
    prepare_function_cuda,
    apply_function_cuda,
    apply_function_adjoint_cuda,
#else
    nullptr,
    nullptr,
    nullptr,
#endif
};
}  // namespace

extern "C" OOS_PLUGIN_EXPORT const oos_surface_plugin_v3*
oos_get_surface_plugin_v3(void) {
  return &plugin;
}

extern "C" OOS_PLUGIN_EXPORT const oos_function_operator_v2*
oos_get_function_operator_v2(void) {
  return &function_operator;
}
