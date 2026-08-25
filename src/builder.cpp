#include "oos/builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <iterator>
#include <mutex>
#include <new>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nlohmann/json.hpp>

#include "oos/analytic_geometry.hpp"
#include "oos/geometry.hpp"
#include "oos/function_operator.hpp"
#include "oos/hash.hpp"
#include "oos/physics.hpp"
#include "oos/plugin.hpp"
#include "oos/surface_behavior.hpp"
#include "oos/validation.hpp"

namespace oos {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::vector<std::pair<double, double>> gauss_legendre_unit_interval(
    std::uint32_t count) {
  if (count == 0)
    throw std::runtime_error("Gauss-Legendre quadrature order is zero");
  std::vector<std::pair<double, double>> result(count);
  const std::uint32_t half = (count + 1u) / 2u;
  for (std::uint32_t i = 0; i < half; ++i) {
    double root =
        std::cos(pi * (static_cast<double>(i) + 0.75) /
                 (static_cast<double>(count) + 0.5));
    double derivative = 0.0;
    for (std::uint32_t iteration = 0; iteration < 64; ++iteration) {
      double previous = 1.0;
      double value = root;
      for (std::uint32_t order = 2; order <= count; ++order) {
        const double next =
            ((2.0 * order - 1.0) * root * value -
             (order - 1.0) * previous) /
            order;
        previous = value;
        value = next;
      }
      derivative =
          count * (root * value - previous) / (root * root - 1.0);
      const double updated = root - value / derivative;
      if (std::abs(updated - root) <= 4.0e-16) {
        root = updated;
        break;
      }
      root = updated;
    }
    const double weight =
        1.0 / ((1.0 - root * root) * derivative * derivative);
    result[i] = {(1.0 - root) * 0.5, weight};
    result[count - 1u - i] = {(1.0 + root) * 0.5, weight};
  }
  return result;
}

struct EmbeddedQuadratureNode {
  double unit_node{};
  double high_weight{};
  double low_weight{};
};

const std::vector<EmbeddedQuadratureNode>&
gauss_kronrod_15_31_unit_interval() {
  // QUADPACK DQK31 coefficients. Odd (zero-based) Kronrod abscissae are the
  // embedded 15-point Gauss rule; both rules are mapped from [-1,1] to [0,1].
  static const std::vector<EmbeddedQuadratureNode> nodes = [] {
    constexpr std::array<double, 16> x{
        0.99800229869339706029, 0.98799251802048542849,
        0.96773907567913913426, 0.93727339240070590431,
        0.89726453234408190088, 0.84820658341042721620,
        0.79041850144246593297, 0.72441773136017004742,
        0.65099674129741697053, 0.57097217260853884754,
        0.48508186364023968069, 0.39415134707756336990,
        0.29918000715316881217, 0.20119409399743452230,
        0.10114206691871749903, 0.0};
    constexpr std::array<double, 16> kronrod{
        0.00537747987292334899, 0.01500794732931612254,
        0.02546084732671532019, 0.03534636079137584622,
        0.04458975132476487661, 0.05348152469092808727,
        0.06200956780067064029, 0.06985412131872825871,
        0.07684968075772037889, 0.08308050282313302104,
        0.08856444305621177065, 0.09312659817082532123,
        0.09664272698362367851, 0.09917359872179195933,
        0.10076984552387559504, 0.10133000701479154902};
    constexpr std::array<double, 8> gauss{
        0.03075324199611726835, 0.07036604748810812471,
        0.10715922046717193501, 0.13957067792615431445,
        0.16626920581699393355, 0.18616100001556221103,
        0.19843148532711157646, 0.20257824192556127288};
    std::vector<EmbeddedQuadratureNode> result;
    result.reserve(31);
    for (std::size_t index = 0; index + 1 < x.size(); ++index) {
      const double low =
          index % 2 == 1 ? 0.5 * gauss[(index - 1) / 2] : 0.0;
      result.push_back(
          {0.5 * (1.0 - x[index]), 0.5 * kronrod[index], low});
      result.push_back(
          {0.5 * (1.0 + x[index]), 0.5 * kronrod[index], low});
    }
    result.push_back({0.5, 0.5 * kronrod.back(),
                      0.5 * gauss.back()});
    return result;
  }();
  return nodes;
}

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 subtract(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 scale(const Vec3& value, double factor) {
  return {factor * value.x, factor * value.y, factor * value.z};
}
double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double triangle_area(const Scene& scene, std::uint32_t primitive) {
  const auto triangle = scene.mesh.triangles.at(primitive);
  return 0.5 * norm(cross(
                   subtract(scene.mesh.vertices[triangle[1]],
                            scene.mesh.vertices[triangle[0]]),
                   subtract(scene.mesh.vertices[triangle[2]],
                            scene.mesh.vertices[triangle[0]])));
}

Vec3 triangle_normal(const Scene& scene, std::uint32_t primitive) {
  const auto triangle = scene.mesh.triangles.at(primitive);
  return normalized(cross(subtract(scene.mesh.vertices[triangle[1]],
                                   scene.mesh.vertices[triangle[0]]),
                          subtract(scene.mesh.vertices[triangle[2]],
                                   scene.mesh.vertices[triangle[0]])));
}

Vec3 triangle_center(const Scene& scene, std::uint32_t primitive) {
  const auto triangle = scene.mesh.triangles.at(primitive);
  return scale(add(add(scene.mesh.vertices[triangle[0]],
                       scene.mesh.vertices[triangle[1]]),
                   scene.mesh.vertices[triangle[2]]),
               1.0 / 3.0);
}

struct WeightedRay {
  Ray ray;
  Stokes stokes;
  Vec3 reference_axis;
  std::int32_t domain{};
  std::uint32_t depth{};
  std::optional<Hit> known_first_hit;
};

template <std::size_t InlineCapacity = 16>
class SmallSparseMap {
 public:
  using Entry = std::pair<std::uint32_t, double>;
  using EntryStorage =
      std::aligned_storage_t<sizeof(Entry), alignof(Entry)>;
  static_assert(std::is_trivially_destructible_v<Entry>);

  SmallSparseMap() noexcept = default;
  SmallSparseMap(const SmallSparseMap& other) {
    for (const auto& [index, value] : other) (*this)[index] = value;
  }
  SmallSparseMap& operator=(const SmallSparseMap& other) {
    if (this == &other) return *this;
    clear();
    for (const auto& [index, value] : other) (*this)[index] = value;
    return *this;
  }
  SmallSparseMap(SmallSparseMap&& other) noexcept {
    move_from(std::move(other));
  }
  SmallSparseMap& operator=(SmallSparseMap&& other) noexcept {
    if (this == &other) return *this;
    clear();
    move_from(std::move(other));
    return *this;
  }

  template <bool IsConst>
  class BasicIterator {
   public:
    using Owner = std::conditional_t<IsConst, const SmallSparseMap,
                                     SmallSparseMap>;
    using reference = std::conditional_t<IsConst, const Entry&, Entry&>;
    using value_type = Entry;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    reference operator*() const { return owner_->entry_at(index_); }
    auto operator->() const { return &owner_->entry_at(index_); }
    BasicIterator& operator++() {
      ++index_;
      return *this;
    }
    bool operator==(const BasicIterator& other) const {
      return owner_ == other.owner_ && index_ == other.index_;
    }
    bool operator!=(const BasicIterator& other) const {
      return !(*this == other);
    }

   private:
    friend class SmallSparseMap;
    BasicIterator(Owner* owner, std::size_t index)
        : owner_(owner), index_(index) {}
    Owner* owner_{};
    std::size_t index_{};
  };

  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;

  double& operator[](std::uint32_t key) {
    if (!spilled_) {
      for (std::size_t index = 0; index < inline_size_; ++index)
        if (inline_entry(index).first == key)
          return inline_entry(index).second;
      if (inline_size_ < InlineCapacity) {
        auto* entry = new (&inline_storage_[inline_size_])
            Entry{key, 0.0};
        ++inline_size_;
        return entry->second;
      }
      spill();
    }
    const auto found = spill_index_.find(key);
    if (found != spill_index_.end())
      return spill_entries_[found->second].second;
    const auto index = spill_entries_.size();
    spill_entries_.push_back({key, 0.0});
    spill_index_.emplace(key, index);
    return spill_entries_.back().second;
  }

  bool contains(std::uint32_t key) const {
    if (!spilled_) {
      for (std::size_t index = 0; index < inline_size_; ++index)
        if (inline_entry(index).first == key) return true;
      return false;
    }
    return spill_index_.count(key) != 0;
  }

  double value(std::uint32_t key) const {
    if (!spilled_) {
      for (std::size_t index = 0; index < inline_size_; ++index)
        if (inline_entry(index).first == key)
          return inline_entry(index).second;
      return 0.0;
    }
    const auto found = spill_index_.find(key);
    return found == spill_index_.end()
               ? 0.0
               : spill_entries_[found->second].second;
  }

  std::size_t size() const {
    return spilled_ ? spill_entries_.size() : inline_size_;
  }
  bool empty() const { return size() == 0; }
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, size()); }
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, size()); }

 private:
  Entry& entry_at(std::size_t index) {
    return spilled_ ? spill_entries_[index] : inline_entry(index);
  }
  const Entry& entry_at(std::size_t index) const {
    return spilled_ ? spill_entries_[index] : inline_entry(index);
  }
  Entry& inline_entry(std::size_t index) {
    return *std::launder(
        reinterpret_cast<Entry*>(&inline_storage_[index]));
  }
  const Entry& inline_entry(std::size_t index) const {
    return *std::launder(
        reinterpret_cast<const Entry*>(&inline_storage_[index]));
  }
  void spill() {
    spilled_ = true;
    spill_entries_.reserve(2 * InlineCapacity);
    spill_index_.reserve(2 * InlineCapacity);
    for (std::size_t index = 0; index < inline_size_; ++index) {
      spill_index_.emplace(inline_entry(index).first,
                           spill_entries_.size());
      spill_entries_.push_back(inline_entry(index));
    }
  }

  void clear() {
    inline_size_ = 0;
    spilled_ = false;
    spill_entries_.clear();
    spill_index_.clear();
  }

  void move_from(SmallSparseMap&& other) {
    if (other.spilled_) {
      spilled_ = true;
      spill_entries_ = std::move(other.spill_entries_);
      spill_index_ = std::move(other.spill_index_);
      other.inline_size_ = 0;
      other.spilled_ = false;
      return;
    }
    for (const auto& [index, value] : other) (*this)[index] = value;
    other.inline_size_ = 0;
  }

  std::array<EntryStorage, InlineCapacity> inline_storage_;
  std::size_t inline_size_{};
  bool spilled_{};
  std::vector<Entry> spill_entries_;
  std::unordered_map<std::uint32_t, std::size_t> spill_index_;
};

struct RowAccumulation {
  SmallSparseMap<> transition;
  SmallSparseMap<> detection;
  SmallSparseMap<> losses;
};

std::map<std::uint32_t, double> ordered_map(
    const SmallSparseMap<>& source) {
  std::map<std::uint32_t, double> result;
  for (const auto& [index, value] : source)
    if (value != 0.0) result.emplace(index, value);
  return result;
}

class DenseComponentAccumulator {
 public:
  void reset(std::size_t size) {
    if (values_.size() != size) {
      values_.assign(size, 0.0);
      touched_mask_.assign(size, 0);
    } else {
      for (const auto index : touched_) {
        values_[index] = 0.0;
        touched_mask_[index] = 0;
      }
    }
    touched_.clear();
  }

  void add(std::uint32_t index, double value) {
    if (index >= values_.size())
      throw std::runtime_error(
          "source accumulation index is out of range: index=" +
          std::to_string(index) + " size=" +
          std::to_string(values_.size()));
    if (!touched_mask_[index]) {
      touched_mask_[index] = 1;
      touched_.push_back(index);
    }
    values_[index] += value;
  }

  void add(const SmallSparseMap<>& source, double factor = 1.0) {
    for (const auto& [index, value] : source)
      add(index, factor * value);
  }

  void add(const DenseComponentAccumulator& source,
           double factor = 1.0) {
    for (const auto index : source.touched_)
      add(index, factor * source.values_[index]);
  }

  double total() const {
    double result = 0.0;
    for (const auto index : touched_) result += values_[index];
    return result;
  }

  double l1_difference(const DenseComponentAccumulator& other) const {
    if (values_.size() != other.values_.size())
      throw std::runtime_error(
          "source accumulators have incompatible dimensions");
    double result = 0.0;
    for (const auto index : touched_)
      result += std::abs(values_[index] - other.values_[index]);
    for (const auto index : other.touched_)
      if (!touched_mask_[index]) result += std::abs(other.values_[index]);
    return result;
  }

  void scale_values(double factor) {
    for (const auto index : touched_) values_[index] *= factor;
  }

  void append_to(SmallSparseMap<>& target) const {
    for (const auto index : touched_)
      if (values_[index] != 0.0) target[index] += values_[index];
  }

  void store_to(std::vector<double>& target, std::size_t offset) const {
    if (offset + values_.size() > target.size())
      throw std::runtime_error(
          "source output accumulation is out of range");
    for (const auto index : touched_)
      target[offset + index] = values_[index];
  }

 private:
  std::vector<double> values_;
  std::vector<std::uint8_t> touched_mask_;
  std::vector<std::uint32_t> touched_;
};

class DenseRowAccumulator {
 public:
  void reset(std::size_t transition_count, std::size_t detection_count,
             std::size_t loss_count) {
    transition.reset(transition_count);
    detection.reset(detection_count);
    losses.reset(loss_count);
  }

  void add(const RowAccumulation& source, double factor = 1.0) {
    transition.add(source.transition, factor);
    detection.add(source.detection, factor);
    losses.add(source.losses, factor);
  }

  void add(const DenseRowAccumulator& source, double factor = 1.0) {
    transition.add(source.transition, factor);
    detection.add(source.detection, factor);
    losses.add(source.losses, factor);
  }

  void add_transition(std::uint32_t index, double value) {
    if (value != 0.0) transition.add(index, value);
  }
  void add_detection(std::uint32_t index, double value) {
    if (value != 0.0) detection.add(index, value);
  }
  void add_loss(std::uint32_t index, double value) {
    if (value != 0.0) losses.add(index, value);
  }

  double total() const {
    return transition.total() + detection.total() + losses.total();
  }


  double l1_difference(const DenseRowAccumulator& other) const {
    return transition.l1_difference(other.transition) +
           detection.l1_difference(other.detection) +
           losses.l1_difference(other.losses);
  }

  void scale_values(double factor) {
    transition.scale_values(factor);
    detection.scale_values(factor);
    losses.scale_values(factor);
  }

  RowAccumulation sparse() const {
    RowAccumulation result;
    transition.append_to(result.transition);
    detection.append_to(result.detection);
    losses.append_to(result.losses);
    return result;
  }

  void store_to(SourceBatch& batch, std::size_t source_index,
                std::size_t transition_count,
                std::size_t detection_count,
                std::size_t loss_count) const {
    transition.store_to(batch.initial_states,
                        source_index * transition_count);
    detection.store_to(batch.direct_detection,
                       source_index * detection_count);
    losses.store_to(batch.direct_losses, source_index * loss_count);
  }

 private:
  DenseComponentAccumulator transition;
  DenseComponentAccumulator detection;
  DenseComponentAccumulator losses;
};

struct SurfaceEmitter {
  std::uint64_t geometry_key{};
  Vec3 center;
  Vec3 normal;
  Vec3 tangent;
  double area_mm2{};
  std::uint32_t surface_id{};
  std::uint32_t surface_basis_id{};
  std::int32_t minus_domain_id{-1};
  std::int32_t plus_domain_id{-1};
  std::string label;
};

struct LocalSurfaceBasis {
  std::vector<std::vector<SurfaceEmitter>> state_emitters;
  std::unordered_map<std::uint64_t, std::uint32_t> primitive_to_state;
  std::vector<std::string> state_labels;
};

bool owns_lambertian_state(const SurfaceModel& surface, double tolerance) {
  if (surface.kind == SurfaceKind::lambertian)
    return surface.reflectivity > tolerance;
  if (surface.kind == SurfaceKind::custom_local) return true;
  return surface.kind == SurfaceKind::sensitive &&
         surface.remainder == RemainderAction::reflect_lambertian &&
         (1.0 - surface.detection_probability) > tolerance;
}

LocalSurfaceBasis make_local_surface_basis(const Scene& scene) {
  using Key = std::tuple<std::uint32_t, std::uint32_t, bool>;
  std::map<Key, std::uint32_t> key_to_state;
  LocalSurfaceBasis result;
  const auto append =
      [&](SurfaceEmitter emitter, bool primary_is_minus) {
        const Key key{emitter.surface_id, emitter.surface_basis_id,
                      primary_is_minus};
        auto [entry, inserted] =
            key_to_state.emplace(key, result.state_emitters.size());
        if (inserted) result.state_emitters.emplace_back();
        const auto state = entry->second;
        result.primitive_to_state.emplace(emitter.geometry_key, state);
        result.state_emitters.at(state).push_back(std::move(emitter));
      };
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    if (!scene.mesh.triangle_transport.empty() &&
        !scene.mesh.triangle_transport.at(primitive))
      continue;
    const auto& surface = scene.surfaces.at(scene.mesh.surface_id[primitive]);
    const bool primary_is_minus =
        scene.mesh.minus_domain_id[primitive] == scene.primary_domain;
    const bool primary_is_plus =
        scene.mesh.plus_domain_id[primitive] == scene.primary_domain;
    if (!owns_lambertian_state(surface, scene.numerics.energy_tolerance) ||
        (!primary_is_minus && !primary_is_plus))
      continue;
    const std::uint32_t basis_id =
        scene.mesh.surface_basis_id.empty()
            ? primitive
            : scene.mesh.surface_basis_id.at(primitive);
    const auto triangle = scene.mesh.triangles.at(primitive);
    append(
        {static_cast<std::uint64_t>(primitive),
         triangle_center(scene, primitive),
         triangle_normal(scene, primitive),
         normalized(subtract(scene.mesh.vertices.at(triangle[1]),
                             scene.mesh.vertices.at(triangle[0]))),
         triangle_area(scene, primitive),
         surface.id,
         basis_id,
         scene.mesh.minus_domain_id.at(primitive),
         scene.mesh.plus_domain_id.at(primitive),
         "primitive:" + std::to_string(primitive)},
        primary_is_minus);
  }
  for (std::uint32_t element_index = 0;
       element_index < scene.mesh.analytic_surface_elements.size();
       ++element_index) {
    const auto& element =
        scene.mesh.analytic_surface_elements.at(element_index);
    const auto& primitive =
        scene.mesh.analytic_primitives.at(element.primitive_index);
    const auto& surface = scene.surfaces.at(primitive.surface_id);
    const bool primary_is_minus =
        primitive.minus_domain_id == scene.primary_domain;
    const bool primary_is_plus =
        primitive.plus_domain_id == scene.primary_domain;
    if (!owns_lambertian_state(surface, scene.numerics.energy_tolerance) ||
        (!primary_is_minus && !primary_is_plus))
      continue;
    Vec3 tangent = primitive.axis_x;
    if (element.coordinates ==
        AnalyticSurfaceCoordinates::cylinder_phi_z) {
      const double phi =
          0.5 * (element.bounds[0] + element.bounds[1]);
      tangent = normalized(
          add(scale(primitive.axis_x, -std::sin(phi)),
              scale(primitive.axis_y, std::cos(phi))));
    }
    append(
        {analytic_surface_element_geometry_key(element_index),
         element.center_mm,
         element.normal,
         tangent,
         element.area_mm2,
         primitive.surface_id,
         element.surface_basis_id,
         primitive.minus_domain_id,
         primitive.plus_domain_id,
         "element:" + std::to_string(element.surface_element)},
        primary_is_minus);
  }
  result.state_labels.reserve(result.state_emitters.size());
  for (const auto& emitters : result.state_emitters) {
    const auto& first = emitters.front();
    const auto& surface = scene.surfaces.at(first.surface_id);
    if (emitters.size() == 1)
      result.state_labels.push_back(surface.name + "/" + first.label);
    else
      result.state_labels.push_back(
          surface.name + "/basis:" +
          std::to_string(first.surface_basis_id));
  }
  return result;
}

struct CustomSurfaceRuntime {
  std::shared_ptr<SurfacePlugin> plugin;
  std::string config_json;
  PluginBuildResult payload;
  std::uint64_t state_offset{};
  std::uint64_t state_count{};
  std::int32_t nonlocal_domain_id{-1};
  bool functional{};
  oos_function_operator_descriptor_v2 function_descriptor{};
};

using CustomRuntimeMap =
    std::unordered_map<std::uint32_t, CustomSurfaceRuntime>;

class StructuredAnalyticIntersector {
 public:
  StructuredAnalyticIntersector(const Scene& scene, const Geometry& geometry)
      : scene_(scene), geometry_(geometry) {
    const auto& primitives = scene.mesh.analytic_primitives;
    for (std::uint32_t index = 0; index < primitives.size(); ++index) {
      const auto& primitive = primitives[index];
      if (primitive.kind != GeometryPrimitiveKind::perforated_disk ||
          primitive.holes.empty())
        continue;
      Aperture aperture;
      aperture.primitive = index;
      aperture.maximum_radius = 0.0;
      aperture.stacks.resize(primitive.holes.size());
      for (const auto& hole : primitive.holes)
        aperture.maximum_radius =
            std::max(aperture.maximum_radius, hole.radius_mm);
      aperture.cell_size = std::max(2.0 * aperture.maximum_radius,
                                    scene.numerics.geometry_tolerance_mm);
      for (std::uint32_t hole = 0; hole < primitive.holes.size(); ++hole) {
        const auto& center = primitive.holes[hole].center_uv_mm;
        aperture.cells[cell(center.x, center.y, aperture.cell_size)]
            .push_back(hole);
      }
      apertures_.push_back(std::move(aperture));
    }

    std::unordered_set<std::uint64_t> assigned;
    for (std::size_t aperture_index = 0;
         aperture_index < apertures_.size(); ++aperture_index) {
      auto& aperture = apertures_[aperture_index];
      const auto& plane = primitives.at(aperture.primitive);
      for (std::uint32_t index = 0; index < primitives.size(); ++index) {
        if (index == aperture.primitive) continue;
        const auto& primitive = primitives[index];
        if (primitive.kind != GeometryPrimitiveKind::disk &&
            primitive.kind != GeometryPrimitiveKind::finite_cylinder)
          continue;
        const Vec3 relative = subtract(primitive.center_mm, plane.center_mm);
        const double u = dot(relative, plane.axis_x);
        const double v = dot(relative, plane.axis_y);
        const auto hole = locate(aperture, u, v);
        if (!hole) continue;
        const auto& expected = plane.holes.at(*hole);
        const double radius = primitive.parameters[0];
        const double scale = std::max({1.0, radius, expected.radius_mm});
        if (std::abs(radius - expected.radius_mm) >
            32.0 * scene.numerics.geometry_tolerance_mm * scale)
          continue;
        for (const auto domain : {primitive.minus_domain_id,
                                  primitive.plus_domain_id}) {
          if (!scene.media.count(domain)) continue;
          aperture.stacks[*hole][domain].push_back(index);
          assigned.insert(domain_primitive_key(domain, index));
          apertures_by_domain_[domain].push_back(aperture_index);
        }
      }
      for (const auto domain : {plane.minus_domain_id,
                                plane.plus_domain_id})
        if (scene.media.count(domain))
          apertures_by_domain_[domain].push_back(aperture_index);
    }
    for (auto& [domain, indices] : apertures_by_domain_) {
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
      (void)domain;
    }

    for (std::uint32_t index = 0; index < primitives.size(); ++index) {
      const auto& primitive = primitives[index];
      for (const auto domain : {primitive.minus_domain_id,
                                primitive.plus_domain_id}) {
        if (!scene.media.count(domain) ||
            assigned.count(domain_primitive_key(domain, index)))
          continue;
        global_by_domain_[domain].push_back(index);
      }
    }
  }

  Hit intersect(const Ray& ray, std::int32_t domain) const {
    thread_local std::vector<std::uint32_t> candidates;
    candidates.clear();
    const auto append = [&](std::uint32_t primitive) {
      if (std::find(candidates.begin(), candidates.end(), primitive) ==
          candidates.end())
        candidates.push_back(primitive);
    };
    if (const auto found = global_by_domain_.find(domain);
        found != global_by_domain_.end())
      for (const auto primitive : found->second) append(primitive);

    if (const auto found = apertures_by_domain_.find(domain);
        found != apertures_by_domain_.end()) {
      const Vec3 direction = normalized(ray.direction);
      for (const auto aperture_index : found->second) {
        const auto& aperture = apertures_.at(aperture_index);
        const auto& plane =
            scene_.mesh.analytic_primitives.at(aperture.primitive);
        std::array<std::optional<std::uint32_t>, 2> holes;
        const Vec3 origin_relative = subtract(ray.origin, plane.center_mm);
        holes[0] = locate(aperture, dot(origin_relative, plane.axis_x),
                          dot(origin_relative, plane.axis_y));
        const double denominator = dot(direction, plane.axis_z);
        if (std::abs(denominator) >
            64.0 * std::numeric_limits<double>::epsilon()) {
          const double distance =
              dot(subtract(plane.center_mm, ray.origin), plane.axis_z) /
              denominator;
          if (distance >= ray.t_min && distance <= ray.t_max) {
            const Vec3 point = add(ray.origin, scale(direction, distance));
            const Vec3 relative = subtract(point, plane.center_mm);
            holes[1] = locate(aperture, dot(relative, plane.axis_x),
                              dot(relative, plane.axis_y));
          }
        }
        for (const auto hole : holes) {
          if (!hole) continue;
          const auto stack = aperture.stacks[*hole].find(domain);
          if (stack == aperture.stacks[*hole].end()) continue;
          for (const auto primitive : stack->second) append(primitive);
        }
      }
    }

    std::optional<Hit> closest;
    for (const auto primitive : candidates) {
      const auto hit = geometry_.intersect_declared_analytic(
          primitive, ray, domain);
      if (hit && (!closest || hit->distance < closest->distance))
        closest = hit;
    }
    // A structured scene can still contain an explicitly unstructured domain
    // or a ray on a topology seam. Preserve correctness with a fail-closed BVH
    // fallback; fully declared structured rays do not take this branch.
    return closest ? *closest : geometry_.intersect(ray, domain);
  }

 private:
  using Cell = std::pair<std::int64_t, std::int64_t>;
  struct Aperture {
    std::uint32_t primitive{};
    double maximum_radius{};
    double cell_size{1.0};
    std::map<Cell, std::vector<std::uint32_t>> cells;
    std::vector<std::unordered_map<std::int32_t,
                                   std::vector<std::uint32_t>>>
        stacks;
  };

  static Cell cell(double u, double v, double size) {
    return {static_cast<std::int64_t>(std::floor(u / size)),
            static_cast<std::int64_t>(std::floor(v / size))};
  }
  static std::uint64_t domain_primitive_key(std::int32_t domain,
                                             std::uint32_t primitive) {
    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(domain))
            << 32u) |
           primitive;
  }
  std::optional<std::uint32_t> locate(const Aperture& aperture, double u,
                                      double v) const {
    const auto base = cell(u, v, aperture.cell_size);
    const auto& plane =
        scene_.mesh.analytic_primitives.at(aperture.primitive);
    std::optional<std::uint32_t> result;
    double best = std::numeric_limits<double>::infinity();
    for (std::int64_t du = -1; du <= 1; ++du)
      for (std::int64_t dv = -1; dv <= 1; ++dv) {
        const auto found =
            aperture.cells.find({base.first + du, base.second + dv});
        if (found == aperture.cells.end()) continue;
        for (const auto index : found->second) {
          const auto& hole = plane.holes.at(index);
          const double d2 =
              (u - hole.center_uv_mm.x) * (u - hole.center_uv_mm.x) +
              (v - hole.center_uv_mm.y) * (v - hole.center_uv_mm.y);
          if (d2 <= hole.radius_mm * hole.radius_mm && d2 < best) {
            best = d2;
            result = index;
          }
        }
      }
    return result;
  }

  const Scene& scene_;
  const Geometry& geometry_;
  std::vector<Aperture> apertures_;
  std::unordered_map<std::int32_t, std::vector<std::size_t>>
      apertures_by_domain_;
  std::unordered_map<std::int32_t, std::vector<std::uint32_t>>
      global_by_domain_;
};

oos_surface_hit_v3 plugin_hit(const Scene& scene, const SurfaceModel& surface,
                              const WeightedRay& ray, const Hit& hit,
                              const Vec3& point,
                              std::int32_t transmitted_domain) {
  oos_surface_hit_v3 result{};
  result.surface_element = hit.surface_element;
  std::copy(hit.barycentric.begin(), hit.barycentric.end(),
            result.barycentric);
  result.incident_side =
      ray.domain == hit.minus_domain_id ? 0u : 1u;
  const Vec3 local_point = subtract(point, surface.frame_origin_mm);
  result.point_local_mm[0] = dot(local_point, surface.frame_x);
  result.point_local_mm[1] = dot(local_point, surface.frame_y);
  result.point_local_mm[2] = dot(local_point, surface.frame_z);
  const auto direction = normalized(ray.ray.direction);
  result.direction_local[0] = dot(direction, surface.frame_x);
  result.direction_local[1] = dot(direction, surface.frame_y);
  result.direction_local[2] = dot(direction, surface.frame_z);
  result.reference_axis_local[0] =
      dot(ray.reference_axis, surface.frame_x);
  result.reference_axis_local[1] =
      dot(ray.reference_axis, surface.frame_y);
  result.reference_axis_local[2] =
      dot(ray.reference_axis, surface.frame_z);
  result.stokes[0] = ray.stokes.i;
  result.stokes[1] = ray.stokes.q;
  result.stokes[2] = ray.stokes.u;
  result.stokes[3] = ray.stokes.v;
  result.incident_refractive_index =
      scene.media.at(ray.domain).refractive_index;
  result.transmitted_refractive_index =
      transmitted_domain >= 0
          ? scene.media.at(transmitted_domain).refractive_index
          : result.incident_refractive_index;
  return result;
}

double signed_basis_angle(const Vec3& from, const Vec3& to,
                          const Vec3& direction) {
  return std::atan2(dot(cross(from, to), direction), dot(from, to));
}

void apply_local_branches(
    const Scene& scene, std::uint64_t geometry_key, std::int32_t channel,
    const Vec3& point,
    const Vec3& direction, const Vec3& toward_incident, const Vec3& s_axis,
    std::int32_t incident_domain, std::int32_t transmitted_domain,
    std::uint32_t depth, double nudge,
    const std::unordered_map<std::uint64_t, std::uint32_t>&
        primitive_to_state,
    const std::unordered_map<std::int32_t, std::uint32_t>& channel_to_column,
    const std::vector<LocalSurfaceBranch>& branches,
    std::vector<WeightedRay>& pending, RowAccumulation& output) {
  for (const auto& branch : branches) {
    switch (branch.kind) {
      case LocalBranchKind::specular_reflection: {
        const Vec3 reflected = reflect(direction, toward_incident);
        pending.push_back(
            {{add(point, scale(toward_incident, nudge)), reflected},
             branch.stokes, s_axis, incident_domain, depth + 1,
             std::nullopt});
        break;
      }
      case LocalBranchKind::specular_transmission: {
        if (transmitted_domain < 0)
          throw std::runtime_error("surface transmits outside the scene");
        const auto transmitted = refract(
            direction, toward_incident,
            scene.media.at(incident_domain).refractive_index,
            scene.media.at(transmitted_domain).refractive_index);
        if (!transmitted)
          throw std::runtime_error(
              "surface requested transmission during total internal "
              "reflection");
        pending.push_back(
            {{add(point, scale(toward_incident, -nudge)), *transmitted},
             branch.stokes, s_axis, transmitted_domain, depth + 1,
             std::nullopt});
        break;
      }
      case LocalBranchKind::straight_transmission:
        if (transmitted_domain < 0) {
          output.losses[2] += branch.stokes.i;
        } else {
          pending.push_back(
              {{add(point, scale(toward_incident, -nudge)), direction},
               branch.stokes, s_axis, transmitted_domain, depth + 1,
               std::nullopt});
        }
        break;
      case LocalBranchKind::lambertian_reflection: {
        const auto state = primitive_to_state.find(geometry_key);
        if (state == primitive_to_state.end())
          throw std::runtime_error(
              "Lambertian branch has no geometry-owned state: geometry_key=" +
              std::to_string(geometry_key));
        output.transition[state->second] += branch.stokes.i;
        break;
      }
      case LocalBranchKind::detection: {
        if (channel < 0)
          throw std::runtime_error("detecting surface has no channel");
        output.detection[channel_to_column.at(channel)] += branch.stokes.i;
        break;
      }
      case LocalBranchKind::absorption:
        output.losses[1] += branch.stokes.i;
        break;
    }
  }
}

void trace_branch(const Scene& scene, const Geometry& geometry,
                  const std::unordered_map<std::uint64_t, std::uint32_t>&
                      primitive_to_state,
                  const std::unordered_map<std::int32_t, std::uint32_t>&
                      channel_to_column,
                  const CustomRuntimeMap& custom_runtimes,
                  const WeightedRay& input, RowAccumulation& output,
                  const StructuredAnalyticIntersector* structured = nullptr) {
  thread_local std::vector<WeightedRay> pending;
  pending.clear();
  if (pending.capacity() < scene.numerics.maximum_specular_hits + 1)
    pending.reserve(scene.numerics.maximum_specular_hits + 1);
  pending.push_back(input);
  while (!pending.empty()) {
    WeightedRay ray = pending.back();
    pending.pop_back();
    if (ray.stokes.i <= scene.numerics.energy_tolerance) {
      output.losses[3] += ray.stokes.i;
      continue;
    }
    if (ray.depth >= scene.numerics.maximum_specular_hits) {
      output.losses[3] += ray.stokes.i;
      continue;
    }
    const Hit hit =
        ray.known_first_hit
            ? *ray.known_first_hit
            : structured ? structured->intersect(ray.ray, ray.domain)
                         : geometry.intersect(ray.ray, ray.domain);
    if (!hit.valid) {
      // Scene validation has already established that the traced domain is
      // enclosed.  A no-hit result is therefore kept separate from physical
      // transmission to the exterior: Embree traces float32 rays, so grazing
      // seams can lose a small amount of weight even for a closed float64
      // input mesh.
      output.losses[4] += ray.stokes.i;
      continue;
    }
    const auto& surface = scene.surfaces.at(hit.surface_id);
    const auto primitive = hit.primitive_id;
    const Vec3 point =
        add(ray.ray.origin, scale(normalized(ray.ray.direction), hit.distance));
    const auto& propagation_medium = scene.media.at(ray.domain);
    const double survival =
        std::isfinite(propagation_medium.absorption_length_mm)
            ? std::exp(-hit.distance /
                       propagation_medium.absorption_length_mm)
            : 1.0;
    output.losses[0] += ray.stokes.i * (1.0 - survival);
    ray.stokes = {ray.stokes.i * survival, ray.stokes.q * survival,
                  ray.stokes.u * survival, ray.stokes.v * survival};
    if (surface.kind == SurfaceKind::specular_reflector ||
        surface.kind == SurfaceKind::lambertian ||
        surface.kind == SurfaceKind::sensitive) {
      const auto minus = hit.minus_domain_id;
      const auto plus = hit.plus_domain_id;
      const bool incident_from_minus = ray.domain == minus;
      if (!incident_from_minus && ray.domain != plus)
        throw std::runtime_error(
            "ray domain is not adjacent to surface: domain=" +
            std::to_string(ray.domain) +
            " surface=" + std::to_string(surface.id) +
            " primitive=" + std::to_string(primitive) +
            " minus=" + std::to_string(minus) +
            " plus=" + std::to_string(plus) +
            " origin=(" + std::to_string(ray.ray.origin.x) + "," +
            std::to_string(ray.ray.origin.y) + "," +
            std::to_string(ray.ray.origin.z) + ")" +
            " direction=(" + std::to_string(ray.ray.direction.x) + "," +
            std::to_string(ray.ray.direction.y) + "," +
            std::to_string(ray.ray.direction.z) + ")");
      const auto transmitted_domain = incident_from_minus ? plus : minus;
      const auto normal = hit.normal;
      const auto toward_incident =
          incident_from_minus ? scale(normal, -1.0) : normal;
      const auto direction = normalized(ray.ray.direction);
      const auto branches = evaluate_builtin_surface(
          surface, {ray.stokes,
                    scene.media.at(ray.domain).refractive_index,
                    transmitted_domain >= 0
                        ? scene.media.at(transmitted_domain).refractive_index
                        : scene.media.at(ray.domain).refractive_index,
                    std::max(0.0, -dot(direction, toward_incident))});
      apply_local_branches(
          scene, hit.geometry_key, hit.channel_id, point, direction,
          toward_incident,
          ray.reference_axis, ray.domain, transmitted_domain, ray.depth,
          geometry.ray_origin_offset_mm(), primitive_to_state,
          channel_to_column, branches, pending, output);
      continue;
    }
    if (surface.kind == SurfaceKind::custom_local) {
      const auto runtime = custom_runtimes.find(surface.id);
      if (runtime == custom_runtimes.end())
        throw std::runtime_error("local custom surface has no runtime");
      const auto minus = hit.minus_domain_id;
      const auto plus = hit.plus_domain_id;
      const std::int32_t transmitted_domain =
          ray.domain == minus ? plus : minus;
      const Vec3 normal = hit.normal;
      const auto interaction = runtime->second.plugin->interact_local(
          runtime->second.config_json, scene.energy_eV,
          plugin_hit(scene, surface, ray, hit, point, transmitted_domain));
      const Vec3 direction = normalized(ray.ray.direction);
      const Vec3 toward_incident =
          ray.domain == minus ? scale(normal, -1.0) : normal;
      const auto branches = evaluate_custom_local_surface(
          interaction,
          {ray.stokes, scene.media.at(ray.domain).refractive_index,
           transmitted_domain >= 0
               ? scene.media.at(transmitted_domain).refractive_index
               : scene.media.at(ray.domain).refractive_index,
           std::max(0.0, -dot(direction, toward_incident))},
          scene.numerics.energy_tolerance);
      apply_local_branches(
          scene, hit.geometry_key, hit.channel_id, point, direction,
          toward_incident,
          ray.reference_axis, ray.domain, transmitted_domain, ray.depth,
          geometry.ray_origin_offset_mm(), primitive_to_state,
          channel_to_column, branches, pending, output);
      continue;
    }
    if (surface.kind == SurfaceKind::custom_nonlocal) {
      const auto runtime = custom_runtimes.find(surface.id);
      if (runtime == custom_runtimes.end())
        throw std::runtime_error("nonlocal custom surface has no runtime");
      const auto minus = hit.minus_domain_id;
      const auto plus = hit.plus_domain_id;
      const bool incident_from_minus = ray.domain == minus;
      if (!incident_from_minus && ray.domain != plus)
        throw std::runtime_error("ray domain is not adjacent to custom interface");
      const std::int32_t transmitted_domain =
          incident_from_minus ? plus : minus;
      if (transmitted_domain < 0)
        throw std::runtime_error(
            "nonlocal custom surface must separate two media");
      if (transmitted_domain != runtime->second.nonlocal_domain_id)
        throw std::runtime_error(
            "generic rays may enter, but may not originate inside, a "
            "nonlocal custom domain");
      const Vec3 normal = hit.normal;
      const Vec3 toward_incident =
          incident_from_minus ? scale(normal, -1.0) : normal;
      const Vec3 direction = normalized(ray.ray.direction);
      Vec3 s_axis = cross(direction, toward_incident);
      if (norm(s_axis) < 1e-14) s_axis = ray.reference_axis;
      s_axis = normalized(s_axis);
      const Stokes interface_basis =
          rotate_stokes(ray.stokes,
                        signed_basis_angle(ray.reference_axis, s_axis, direction));
      SurfaceModel dielectric;
      dielectric.kind = SurfaceKind::dielectric_fresnel;
      const auto interface_branches = evaluate_builtin_surface(
          dielectric,
          {interface_basis,
           scene.media.at(ray.domain).refractive_index,
           scene.media.at(transmitted_domain).refractive_index,
           std::max(0.0, -dot(direction, toward_incident))});
      for (const auto& branch : interface_branches) {
        if (branch.kind == LocalBranchKind::specular_reflection) {
          const Vec3 reflected_direction = reflect(direction, toward_incident);
          pending.push_back(
              {{add(point, scale(toward_incident,
                                 geometry.ray_origin_offset_mm())),
                reflected_direction},
               branch.stokes, s_axis, ray.domain, ray.depth + 1,
               std::nullopt});
          continue;
        }
        if (branch.kind != LocalBranchKind::specular_transmission)
          throw std::runtime_error(
              "dielectric entry behavior returned an invalid branch");
        const auto transmitted_direction =
            refract(direction, toward_incident,
                    scene.media.at(ray.domain).refractive_index,
                    scene.media.at(transmitted_domain).refractive_index);
        if (transmitted_direction &&
            branch.stokes.i > scene.numerics.energy_tolerance) {
          WeightedRay transmitted_ray{
              {add(point, scale(toward_incident,
                                -geometry.ray_origin_offset_mm())),
              *transmitted_direction},
              branch.stokes, s_axis, transmitted_domain, ray.depth + 1,
              std::nullopt};
          const auto deposition = runtime->second.plugin->deposit_nonlocal(
              runtime->second.config_json, scene.energy_eV,
              plugin_hit(scene, surface, transmitted_ray, hit, point,
                         transmitted_domain));
          if (deposition.weights.empty())
            throw std::runtime_error(
                "nonlocal custom surface returned an empty deposition");
          double deposited = 0.0;
          for (std::size_t i = 0; i < deposition.weights.size(); ++i) {
            const auto weight = deposition.weights[i];
            const auto state = deposition.state_indices[i];
            if (!std::isfinite(weight) || weight < 0.0 ||
                state >= runtime->second.state_count)
              throw std::runtime_error(
                  "nonlocal custom surface returned an invalid deposition");
            deposited += weight;
            output.transition[static_cast<std::uint32_t>(
                runtime->second.state_offset + state)] +=
                branch.stokes.i * weight;
          }
          if (std::abs(deposited - 1.0) >
              scene.numerics.energy_tolerance)
            throw std::runtime_error(
                "nonlocal custom surface deposition is not conservative");
        }
      }
      continue;
    }

    const auto minus = hit.minus_domain_id;
    const auto plus = hit.plus_domain_id;
    const bool incident_from_minus = ray.domain == minus;
    if (!incident_from_minus && ray.domain != plus)
      throw std::runtime_error("ray domain is not adjacent to interface");
    const std::int32_t transmitted_domain =
        incident_from_minus ? plus : minus;
    if (transmitted_domain < 0) {
      output.losses[2] += ray.stokes.i;
      continue;
    }
    const Vec3 geometric_normal = hit.normal;
    const Vec3 toward_incident =
        incident_from_minus ? scale(geometric_normal, -1.0) : geometric_normal;
    const Vec3 direction = normalized(ray.ray.direction);
    Vec3 s_axis = cross(direction, toward_incident);
    if (norm(s_axis) < 1e-14) s_axis = ray.reference_axis;
    s_axis = normalized(s_axis);
    const Stokes interface_basis =
        rotate_stokes(ray.stokes,
                      signed_basis_angle(ray.reference_axis, s_axis, direction));
    const auto& incident_medium = scene.media.at(ray.domain);
    const auto& transmitted_medium = scene.media.at(transmitted_domain);
    const double cos_i = std::max(0.0, -dot(direction, toward_incident));
    const auto branches = evaluate_builtin_surface(
        surface, {interface_basis, incident_medium.refractive_index,
                  transmitted_medium.refractive_index, cos_i});
    apply_local_branches(
        scene, hit.geometry_key, hit.channel_id, point, direction,
        toward_incident, s_axis,
        ray.domain, transmitted_domain, ray.depth,
        geometry.ray_origin_offset_mm(), primitive_to_state, channel_to_column,
        branches, pending, output);
  }
}

double row_total(const RowAccumulation& row) {
  double result = 0.0;
  for (const auto& [column, value] : row.transition) {
    (void)column;
    result += value;
  }
  for (const auto& [column, value] : row.detection) {
    (void)column;
    result += value;
  }
  for (const auto& [column, value] : row.losses) {
    (void)column;
    result += value;
  }
  return result;
}

void add_row(RowAccumulation& target, const RowAccumulation& source,
             double factor = 1.0) {
  for (const auto& [column, value] : source.transition)
    target.transition[column] += factor * value;
  for (const auto& [column, value] : source.detection)
    target.detection[column] += factor * value;
  for (const auto& [column, value] : source.losses)
    target.losses[column] += factor * value;
}

double map_l1_difference(const SmallSparseMap<>& first,
                         const SmallSparseMap<>& second) {
  double result = 0.0;
  for (const auto& [column, value] : first)
    result += std::abs(value - second.value(column));
  for (const auto& [column, value] : second)
    if (!first.contains(column)) result += std::abs(value);
  return result;
}

double row_l1_difference(const RowAccumulation& first,
                         const RowAccumulation& second) {
  return map_l1_difference(first.transition, second.transition) +
         map_l1_difference(first.detection, second.detection) +
         map_l1_difference(first.losses, second.losses);
}

Vec3 shape_factor_reference_axis(const Vec3& direction) {
  const Vec3 trial =
      std::abs(direction.z) < 0.9 ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
  return normalized(cross(trial, direction));
}

struct ShapeFactorTriangle {
  Vec3 a;
  Vec3 b;
  Vec3 c;
  std::uint32_t primitive{};
  std::uint32_t depth{};
  // Flags correspond to edges (a,b), (b,c), and (c,a).
  std::array<bool, 3> feature_edges{};
};

double exact_triangle_solid_angle(const Vec3& point,
                                  const ShapeFactorTriangle& triangle) {
  const Vec3 a = subtract(triangle.a, point);
  const Vec3 b = subtract(triangle.b, point);
  const Vec3 c = subtract(triangle.c, point);
  const double la = norm(a);
  const double lb = norm(b);
  const double lc = norm(c);
  if (!(la > 0.0) || !(lb > 0.0) || !(lc > 0.0))
    throw std::runtime_error(
        "shape-factor source lies on a boundary triangle");
  const double numerator = std::abs(dot(a, cross(b, c)));
  const double denominator =
      la * lb * lc + dot(a, b) * lc + dot(b, c) * la +
      dot(c, a) * lb;
  return 2.0 * std::atan2(numerator, denominator);
}

double estimated_triangle_solid_angle(
    const Vec3& displacement, double distance,
    const ShapeFactorTriangle& triangle) {
  const Vec3 twice_area_normal =
      cross(subtract(triangle.b, triangle.a),
            subtract(triangle.c, triangle.a));
  const double distance_squared = distance * distance;
  return 0.5 * std::abs(dot(twice_area_normal, displacement)) /
         (distance_squared * distance);
}

struct ShapeFactorTriangleSample {
  Vec3 direction;
  double solid_angle{};
};

std::array<ShapeFactorTriangle, 4> subdivide_shape_factor_triangle(
    const ShapeFactorTriangle& triangle) {
  const Vec3 ab = scale(add(triangle.a, triangle.b), 0.5);
  const Vec3 bc = scale(add(triangle.b, triangle.c), 0.5);
  const Vec3 ca = scale(add(triangle.c, triangle.a), 0.5);
  const auto next_depth = triangle.depth + 1;
  return {
      ShapeFactorTriangle{
          triangle.a, ab, ca, triangle.primitive, next_depth,
          {triangle.feature_edges[0], false, triangle.feature_edges[2]}},
      ShapeFactorTriangle{
          ab, triangle.b, bc, triangle.primitive, next_depth,
          {triangle.feature_edges[0], triangle.feature_edges[1], false}},
      ShapeFactorTriangle{
          ca, bc, triangle.c, triangle.primitive, next_depth,
          {false, triangle.feature_edges[1], triangle.feature_edges[2]}},
      ShapeFactorTriangle{ab, bc, ca, triangle.primitive, next_depth,
                          {false, false, false}}};
}

using ShapeFactorFeatureEdges = std::vector<std::array<bool, 3>>;

ShapeFactorFeatureEdges build_shape_factor_feature_edges(
    const Scene& scene, double dihedral_degrees) {
  struct EdgeOwner {
    std::uint32_t primitive{};
    std::uint32_t local_edge{};
  };
  const auto edge_key = [](std::uint32_t first, std::uint32_t second) {
    const auto low = std::min(first, second);
    const auto high = std::max(first, second);
    return (static_cast<std::uint64_t>(low) << 32u) | high;
  };
  const auto domain_pair = [&](std::uint32_t primitive) {
    auto first = scene.mesh.minus_domain_id.at(primitive);
    auto second = scene.mesh.plus_domain_id.at(primitive);
    if (second < first) std::swap(first, second);
    return std::pair{first, second};
  };
  ShapeFactorFeatureEdges result(scene.mesh.triangles.size());
  std::unordered_map<std::uint64_t, EdgeOwner> owners;
  owners.reserve(scene.mesh.triangles.size() * 2);
  std::vector<Vec3> normals(scene.mesh.triangles.size());
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive)
    normals[primitive] = triangle_normal(scene, primitive);
  const double minimum_normal_cosine =
      std::cos(dihedral_degrees * pi / 180.0);
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    const auto triangle = scene.mesh.triangles.at(primitive);
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> edges{{
        {triangle[0], triangle[1]},
        {triangle[1], triangle[2]},
        {triangle[2], triangle[0]},
    }};
    for (std::uint32_t local_edge = 0; local_edge < edges.size();
         ++local_edge) {
      const auto key =
          edge_key(edges[local_edge].first, edges[local_edge].second);
      const auto [found, inserted] =
          owners.emplace(key, EdgeOwner{primitive, local_edge});
      if (inserted) continue;
      const auto other = found->second;
      const double normal_cosine =
          std::abs(dot(normals[primitive], normals[other.primitive]));
      const bool feature =
          scene.mesh.surface_id.at(primitive) !=
              scene.mesh.surface_id.at(other.primitive) ||
          domain_pair(primitive) != domain_pair(other.primitive) ||
          normal_cosine < minimum_normal_cosine;
      if (feature) {
        result[primitive][local_edge] = true;
        result[other.primitive][other.local_edge] = true;
      }
    }
  }
  return result;
}

struct ShapeFactorTraceContext {
  const Scene& scene;
  const Geometry& geometry;
  const std::unordered_map<std::uint64_t, std::uint32_t>& primitive_to_state;
  const std::unordered_map<std::int32_t, std::uint32_t>& channel_to_column;
  const CustomRuntimeMap& custom_runtimes;
  const StructuredAnalyticIntersector* structured_intersector;
  const SourcePoint& source;
  const ShapeFactorOptions& options;
};

struct ShapeFactorBoundaryCandidate {
  std::uint32_t index{};
  Vec3 center;
  Vec3 outward;
};

struct ShapeFactorDomainCandidates {
  std::vector<ShapeFactorBoundaryCandidate> triangles;
  std::vector<ShapeFactorBoundaryCandidate> analytic_elements;
};

ShapeFactorTriangleSample sample_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle) {
  const Vec3 center =
      scale(add(add(triangle.a, triangle.b), triangle.c), 1.0 / 3.0);
  const Vec3 displacement = subtract(center, context.source.position);
  const double distance = norm(displacement);
  if (!(distance > 0.0))
    throw std::runtime_error(
        "shape-factor source coincides with a triangle centroid");
  const double estimate =
      estimated_triangle_solid_angle(displacement, distance, triangle);
  const double threshold =
      4.0 * pi *
      context.options.maximum_approximate_solid_angle_fraction;
  const bool require_exact =
      !(threshold > 0.0) || !std::isfinite(estimate) ||
      estimate > threshold;
  return {scale(displacement, 1.0 / distance),
          require_exact
              ? exact_triangle_solid_angle(context.source.position, triangle)
              : estimate};
}

RowAccumulation evaluate_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle,
    const ShapeFactorTriangleSample& sample,
    const Hit* known_first_hit = nullptr) {
  RowAccumulation result;
  Hit first;
  if (known_first_hit) {
    first = *known_first_hit;
  } else {
    const auto missing = std::numeric_limits<std::uint32_t>::max();
    const auto replacement =
        context.scene.mesh.triangle_source_analytic_primitive.empty()
            ? missing
            : context.scene.mesh.triangle_source_analytic_primitive.at(
                  triangle.primitive);
    const bool structured =
        context.options.backend != ShapeFactorBackend::generic_bvh &&
        replacement != missing;
    if (structured) {
      const auto hit = context.geometry.intersect_declared_analytic(
          replacement, {context.source.position, sample.direction},
          context.source.domain);
      if (!hit)
        throw std::runtime_error(
            "structured analytic triangle replacement was not intersected");
      first = *hit;
    } else {
      if (context.options.backend ==
          ShapeFactorBackend::structured_analytic)
        throw std::runtime_error(
            "structured analytic backend requires a triangle source "
            "replacement");
      first = context.geometry.intersect(
          {context.source.position, sample.direction},
          context.source.domain);
    }
  }
  // A non-convex boundary can contain triangles hidden behind the first
  // domain boundary.  They do not own any first-flight solid angle.  Mixed
  // visibility is resolved by the child comparison below.
  const bool exact_triangle =
      first.valid && first.kind == GeometryPrimitiveKind::triangle &&
      first.primitive_id == triangle.primitive;
  const bool analytic_replacement =
      first.valid &&
      !context.scene.mesh.triangle_transport.empty() &&
      !context.scene.mesh.triangle_transport.at(triangle.primitive) &&
      !context.scene.mesh.triangle_source_quadrature.empty() &&
      context.scene.mesh.triangle_source_quadrature.at(
          triangle.primitive) &&
      first.kind != GeometryPrimitiveKind::triangle &&
      first.surface_id ==
          context.scene.mesh.surface_id.at(triangle.primitive) &&
      (first.minus_domain_id == context.source.domain ||
       first.plus_domain_id == context.source.domain);
  if (!exact_triangle && !analytic_replacement)
    return result;
  if (!(sample.solid_angle > 0.0)) return result;
  const double weight =
      context.source.stokes.i * sample.solid_angle / (4.0 * pi);
  trace_branch(
      context.scene, context.geometry, context.primitive_to_state,
      context.channel_to_column, context.custom_runtimes,
      {{context.source.position, sample.direction},
       {weight, 0.0, 0.0, 0.0},
       shape_factor_reference_axis(sample.direction), context.source.domain, 0,
       first},
      result, context.structured_intersector);
  return result;
}

double projected_aperture_coverage(
    const ShapeFactorTraceContext& context,
    const AnalyticSurfaceElement& element, double unmasked_weight) {
  const auto missing = std::numeric_limits<std::uint32_t>::max();
  if (element.projected_aperture_primitive_index == missing &&
      element.projected_aperture_hole_index == missing)
    return 1.0;
  if (element.projected_aperture_primitive_index == missing ||
      element.projected_aperture_hole_index == missing)
    throw std::runtime_error(
        "shape-factor element has an incomplete projected aperture");
  const auto& receiver = context.scene.mesh.analytic_primitives.at(
      element.primitive_index);
  const auto& aperture = context.scene.mesh.analytic_primitives.at(
      element.projected_aperture_primitive_index);
  if (aperture.kind != GeometryPrimitiveKind::perforated_disk)
    throw std::runtime_error(
        "projected aperture primitive must be a perforated disk");
  if (element.projected_aperture_hole_index >= aperture.holes.size())
    throw std::runtime_error(
        "projected aperture hole index is out of range");
  const auto& hole =
      aperture.holes.at(element.projected_aperture_hole_index);
  const Vec3 source = context.source.position;
  const double denominator =
      dot(subtract(receiver.center_mm, source), aperture.axis_z);
  if (std::abs(denominator) <=
      64.0 * std::numeric_limits<double>::epsilon())
    throw std::runtime_error(
        "receiver and aperture projection are degenerate");
  const double fraction =
      dot(subtract(aperture.center_mm, source), aperture.axis_z) /
      denominator;
  const Vec3 hole_center =
      add(aperture.center_mm,
          add(scale(aperture.axis_x, hole.center_uv_mm.x),
              scale(aperture.axis_y, hole.center_uv_mm.y)));
  const auto aperture_coordinates = [&](const Vec3& point) {
    const Vec3 projected =
        add(source, scale(subtract(point, source), fraction));
    const Vec3 relative = subtract(projected, hole_center);
    return Vec2{dot(relative, aperture.axis_x),
                dot(relative, aperture.axis_y)};
  };
  const Vec2 center = aperture_coordinates(element.center_mm);
  const double center_distance =
      std::hypot(center.x, center.y);
  const double center_decision =
      center_distance <= hole.radius_mm ? 1.0 : 0.0;
  // Non-polar elements, notably aperture-tunnel cylinder panels, retain the
  // same center-ray visibility decision as the generic first-hit path. Polar
  // receiver cells additionally support finite-area edge coverage below.
  if (element.coordinates !=
      AnalyticSurfaceCoordinates::annulus_r2_phi)
    return center_decision;
  const double radial_min =
      std::sqrt(std::max(0.0, element.bounds[0]));
  const double radial_max =
      std::sqrt(std::max(0.0, element.bounds[1]));
  const double phi_min = element.bounds[2];
  const double phi_max = element.bounds[3];
  const Vec3 receiver_relative =
      subtract(element.center_mm, receiver.center_mm);
  const Vec2 sample{
      dot(receiver_relative, receiver.axis_x),
      dot(receiver_relative, receiver.axis_y)};
  double cell_radius = 0.0;
  for (const double radius : {radial_min, radial_max})
    for (const double phi : {phi_min, phi_max}) {
      const Vec2 corner{radius * std::cos(phi),
                        radius * std::sin(phi)};
      cell_radius =
          std::max(cell_radius,
                   std::hypot(corner.x - sample.x,
                              corner.y - sample.y));
    }
  const double uncertainty = std::abs(fraction) * cell_radius;
  if (center_distance + uncertainty <= hole.radius_mm) return 1.0;
  if (center_distance - uncertainty >= hole.radius_mm) return 0.0;
  if (unmasked_weight <
      context.options.aperture_edge_weight_threshold)
    return center_decision;

  // Parameterize the finite receiver cell around the receiver primitive,
  // then project each radial direction into the aperture plane.  This is the
  // finite-area circular clipping rule, generalized to arbitrary
  // coplanar local frames.
  const Vec2 projected_receiver_center =
      aperture_coordinates(receiver.center_mm);
  double radial_area_integral = 0.0;
  for (const auto& [unit_node, unit_weight] :
       gauss_legendre_unit_interval(
           context.options.aperture_edge_phi_order)) {
    const double phi =
        phi_min + (phi_max - phi_min) * unit_node;
    const Vec3 radial_direction =
        add(scale(receiver.axis_x, std::cos(phi)),
            scale(receiver.axis_y, std::sin(phi)));
    const Vec2 projected_direction{
        fraction * dot(radial_direction, aperture.axis_x),
        fraction * dot(radial_direction, aperture.axis_y)};
    const double quadratic =
        projected_direction.x * projected_direction.x +
        projected_direction.y * projected_direction.y;
    const double linear =
        2.0 * (projected_receiver_center.x *
                   projected_direction.x +
               projected_receiver_center.y *
                   projected_direction.y);
    const double constant =
        projected_receiver_center.x * projected_receiver_center.x +
        projected_receiver_center.y * projected_receiver_center.y -
        hole.radius_mm * hole.radius_mm;
    const double discriminant =
        linear * linear - 4.0 * quadratic * constant;
    if (!(quadratic > 0.0) || !(discriminant > 0.0)) continue;
    const double root = std::sqrt(discriminant);
    const double lower_root =
        (-linear - root) / (2.0 * quadratic);
    const double upper_root =
        (-linear + root) / (2.0 * quadratic);
    const double lower = std::max({radial_min, lower_root, 0.0});
    const double upper = std::min(radial_max, upper_root);
    if (upper > lower)
      radial_area_integral +=
          unit_weight * (upper * upper - lower * lower);
  }
  const double covered_area =
      0.5 * (phi_max - phi_min) * radial_area_integral;
  const double cell_area =
      0.5 * (radial_max * radial_max -
             radial_min * radial_min) *
      (phi_max - phi_min);
  if (!(cell_area > 0.0))
    throw std::runtime_error(
        "projected aperture cell has zero area");
  return std::clamp(covered_area / cell_area, 0.0, 1.0);
}

bool accumulate_simple_structured_hit(
    const ShapeFactorTraceContext& context, const Hit& hit,
    const Vec3& direction, double weight, DenseRowAccumulator& output) {
  if (weight <= context.scene.numerics.energy_tolerance) {
    output.add_loss(3, weight);
    return true;
  }
  const auto& surface = context.scene.surfaces.at(hit.surface_id);
  if (surface.kind != SurfaceKind::lambertian &&
      surface.kind != SurfaceKind::sensitive)
    return false;
  if (surface.kind == SurfaceKind::sensitive &&
      surface.remainder != RemainderAction::absorb &&
      surface.remainder != RemainderAction::reflect_lambertian)
    return false;
  const auto& medium = context.scene.media.at(context.source.domain);
  const double survival =
      std::isfinite(medium.absorption_length_mm)
          ? std::exp(-hit.distance / medium.absorption_length_mm)
          : 1.0;
  output.add_loss(0, weight * (1.0 - survival));
  const double incident = weight * survival;
  const auto add_lambertian = [&](double value) {
    if (!(value > 0.0)) return;
    const auto state = context.primitive_to_state.find(hit.geometry_key);
    if (state == context.primitive_to_state.end())
      throw std::runtime_error(
          "structured Lambertian hit has no geometry-owned state");
    output.add_transition(state->second, value);
  };
  if (surface.kind == SurfaceKind::lambertian) {
    add_lambertian(incident * surface.reflectivity);
    output.add_loss(1, incident * (1.0 - surface.reflectivity));
    return true;
  }

  const double detected = incident * surface.detection_probability;
  output.add_detection(context.channel_to_column.at(hit.channel_id), detected);
  const double remainder = incident - detected;
  if (surface.remainder == RemainderAction::absorb)
    output.add_loss(1, remainder);
  else if (surface.remainder == RemainderAction::reflect_lambertian)
    add_lambertian(remainder);
  else
    return false;
  (void)direction;
  return true;
}

void accumulate_shape_factor_analytic_element(
    const ShapeFactorTraceContext& context, std::uint32_t element_index,
    DenseRowAccumulator& output) {
  RowAccumulation result;
  const auto& element =
      context.scene.mesh.analytic_surface_elements.at(element_index);
  const Vec3 displacement =
      subtract(element.center_mm, context.source.position);
  const double distance = norm(displacement);
  if (!(distance > 0.0))
    throw std::runtime_error(
        "shape-factor source coincides with an analytic surface element");
  const Vec3 direction = scale(displacement, 1.0 / distance);
  const double projected_cosine =
      std::max(0.0, dot(direction, element.normal));
  if (!(projected_cosine > 0.0)) return;
  const double unmasked_weight =
      context.source.stokes.i * projected_cosine * element.area_mm2 /
      (4.0 * pi * distance * distance);
  const double coverage = projected_aperture_coverage(
      context, element, unmasked_weight);
  if (!(coverage > 0.0)) return;
  const auto expected_key =
      analytic_surface_element_geometry_key(element_index);
  const bool structured =
      context.options.backend != ShapeFactorBackend::generic_bvh &&
      element.source_visibility != AnalyticSourceVisibility::ray_traced;
  const auto declared_hit = [&]() {
    const auto& primitive =
        context.scene.mesh.analytic_primitives.at(
            element.primitive_index);
    return Hit{true,
               primitive.kind,
               expected_key,
               element.primitive_index,
               primitive.surface_id,
               distance,
               element.normal,
               primitive.minus_domain_id,
               primitive.plus_domain_id,
               primitive.channel_id,
               element.surface_basis_id,
               element.surface_element,
               {1.0, 0.0, 0.0}};
  };
  Hit first;
  if (structured) {
    // The aperture can hide the center ray while exposing a finite fraction
    // of the receiver cell.  Once the exact covered fraction is known, route
    // that fraction to the receiver surface rather than the blocking plane.
    first = declared_hit();
  } else {
    if (context.options.backend ==
        ShapeFactorBackend::structured_analytic)
      throw std::runtime_error(
          "structured analytic backend requires an analytic-element "
          "visibility declaration");
    first = context.geometry.intersect(
        {context.source.position, direction}, context.source.domain);
    if (element.projected_aperture_primitive_index !=
        std::numeric_limits<std::uint32_t>::max())
      first = declared_hit();
    else if (!first.valid || first.geometry_key != expected_key)
      return;
  }
  const double weight = unmasked_weight * coverage;
  if (!(weight > 0.0) || !std::isfinite(weight))
    throw std::runtime_error(
        "analytic shape-factor element produced invalid weight");
  if (structured && accumulate_simple_structured_hit(
                        context, first, direction, weight, output))
    return;
  trace_branch(
      context.scene, context.geometry, context.primitive_to_state,
      context.channel_to_column, context.custom_runtimes,
      {{context.source.position, direction},
       {weight, 0.0, 0.0, 0.0},
       shape_factor_reference_axis(direction),
       context.source.domain,
       0,
       first},
      result, context.structured_intersector);
  output.add(result);
}

void accumulate_structured_disk_pass(
    const ShapeFactorTraceContext& context, std::uint32_t primitive_index,
    std::uint32_t mu_order, std::uint32_t phi_count,
    DenseRowAccumulator& output,
    DenseRowAccumulator* embedded_low = nullptr) {
  const auto& primitive =
      context.scene.mesh.analytic_primitives.at(primitive_index);
  if (primitive.kind != GeometryPrimitiveKind::disk ||
      primitive.source_integral !=
          AnalyticSourceIntegral::directional_disk)
    throw std::runtime_error(
        "structured directional source integral requires a declared disk");
  const Vec3 relative =
      subtract(context.source.position, primitive.center_mm);
  const double source_u = dot(relative, primitive.axis_x);
  const double source_v = dot(relative, primitive.axis_y);
  const double signed_height = dot(relative, primitive.axis_z);
  const double height = std::abs(signed_height);
  if (!(height > context.scene.numerics.geometry_tolerance_mm))
    throw std::runtime_error(
        "structured disk source lies in the receiver plane");
  const double toward_plane = signed_height > 0.0 ? -1.0 : 1.0;
  const double radius = primitive.parameters[0];
  const double circle_constant =
      source_u * source_u + source_v * source_v - radius * radius;
  const bool embedded = embedded_low != nullptr;
  if (embedded && (mu_order != 31 || phi_count % 2 != 0))
    throw std::runtime_error(
        "embedded structured disk rule requires order 31 and even phi count");
  const auto mu_nodes = embedded
                            ? std::vector<std::pair<double, double>>{}
                            : gauss_legendre_unit_interval(mu_order);

  for (std::uint32_t phi_index = 0; phi_index < phi_count;
       ++phi_index) {
    const double phi =
        2.0 * pi *
        (static_cast<double>(phi_index) + (embedded ? 0.0 : 0.5)) /
        static_cast<double>(phi_count);
    const double cosine = std::cos(phi);
    const double sine = std::sin(phi);
    const double projection = source_u * cosine + source_v * sine;
    const double discriminant =
        projection * projection - circle_constant;
    if (!(discriminant > 0.0)) continue;
    const double root = std::sqrt(discriminant);
    const double radial_min = std::max(0.0, -projection - root);
    const double radial_max = -projection + root;
    if (!(radial_max > radial_min)) continue;
    const double mu_min =
        height / std::hypot(height, radial_max);
    const double mu_max =
        height / std::hypot(height, radial_min);
    if (!(mu_max > mu_min)) continue;
    const Vec3 radial_direction =
        add(scale(primitive.axis_x, cosine),
            scale(primitive.axis_y, sine));
    const auto trace_node = [&](double unit_mu, double unit_weight,
                                double embedded_weight) {
      const double mu = mu_min + (mu_max - mu_min) * unit_mu;
      const double transverse =
          std::sqrt(std::max(0.0, 1.0 - mu * mu));
      const Vec3 direction =
          add(scale(radial_direction, transverse),
              scale(primitive.axis_z, toward_plane * mu));
      const double weight =
          context.source.stokes.i * (mu_max - mu_min) * unit_weight /
          (2.0 * static_cast<double>(phi_count));
      const auto first = context.geometry.intersect_declared_analytic(
          primitive_index, {context.source.position, direction},
          context.source.domain);
      if (!first)
        throw std::runtime_error(
            "structured disk quadrature did not intersect its receiver");
      RowAccumulation node;
      trace_branch(
          context.scene, context.geometry, context.primitive_to_state,
          context.channel_to_column, context.custom_runtimes,
          {{context.source.position, direction},
           {weight, 0.0, 0.0, 0.0},
           shape_factor_reference_axis(direction), context.source.domain, 0,
           *first},
          node, context.structured_intersector);
      output.add(node);
      if (embedded_low && embedded_weight > 0.0 &&
          phi_index % 2u == 0u)
        embedded_low->add(
            node, 2.0 * embedded_weight / unit_weight);
    };
    if (embedded) {
      for (const auto& node : gauss_kronrod_15_31_unit_interval())
        trace_node(node.unit_node, node.high_weight, node.low_weight);
    } else {
      for (const auto& [unit_mu, unit_weight] : mu_nodes)
        trace_node(unit_mu, unit_weight, 0.0);
    }
  }
}

double accumulate_structured_disk(
    const ShapeFactorTraceContext& context, std::uint32_t primitive_index,
    std::size_t transition_count, std::size_t detection_count,
    std::size_t loss_count, DenseRowAccumulator& output) {
  DenseRowAccumulator high;
  high.reset(transition_count, detection_count, loss_count);
  DenseRowAccumulator low;
  low.reset(transition_count, detection_count, loss_count);
  if (context.options.structured_disk_mu_order == 31 &&
      context.options.structured_disk_phi_count % 2u == 0u) {
    accumulate_structured_disk_pass(
        context, primitive_index, 31,
        context.options.structured_disk_phi_count, high, &low);
  } else {
    accumulate_structured_disk_pass(
        context, primitive_index, context.options.structured_disk_mu_order,
        context.options.structured_disk_phi_count, high);
    accumulate_structured_disk_pass(
        context, primitive_index,
        std::max(1u, context.options.structured_disk_mu_order / 2u),
        std::max(1u, context.options.structured_disk_phi_count / 2u), low);
  }
  const double error = high.l1_difference(low);
  output.add(high);
  return error;
}

struct ShapeFactorAssessment {
  std::array<ShapeFactorTriangle, 4> children;
  std::array<ShapeFactorTriangleSample, 4> child_samples;
  std::array<RowAccumulation, 4> child_evaluations;
  RowAccumulation child_sum;
  double error{};
};

ShapeFactorAssessment assess_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle,
    RowAccumulation parent) {
  ShapeFactorAssessment result;
  result.children = subdivide_shape_factor_triangle(triangle);
  for (std::size_t child = 0; child < result.children.size(); ++child) {
    result.child_samples[child] =
        sample_shape_factor_triangle(context, result.children[child]);
    result.child_evaluations[child] = evaluate_shape_factor_triangle(
        context, result.children[child], result.child_samples[child]);
    add_row(result.child_sum, result.child_evaluations[child]);
  }
  result.error = row_l1_difference(parent, result.child_sum);
  return result;
}

struct ShapeFactorIntegral {
  RowAccumulation row;
  double estimated_l1_error{};
};

ShapeFactorIntegral integrate_shape_factor_triangle(
    const ShapeFactorTraceContext& context,
    const ShapeFactorTriangle& triangle, double error_budget,
    double numerical_error_floor,
    std::optional<RowAccumulation> parent_evaluation = std::nullopt,
    std::optional<ShapeFactorTriangleSample> triangle_sample = std::nullopt) {
  const ShapeFactorTriangleSample sample =
      triangle_sample ? std::move(*triangle_sample)
                      : sample_shape_factor_triangle(context, triangle);
  RowAccumulation parent =
      parent_evaluation
          ? std::move(*parent_evaluation)
          : evaluate_shape_factor_triangle(context, triangle, sample);
  const double solid_angle_fraction =
      context.source.stokes.i * sample.solid_angle /
      (4.0 * pi);
  const bool touches_feature =
      std::any_of(triangle.feature_edges.begin(),
                  triangle.feature_edges.end(),
                  [](bool value) { return value; });
  const double refinement_threshold =
      touches_feature
          ? context.options.minimum_feature_solid_angle_fraction
          : context.options.minimum_refinement_solid_angle_fraction;
  if (solid_angle_fraction < refinement_threshold)
    return {std::move(parent), 0.0};

  ShapeFactorAssessment assessment =
      assess_shape_factor_triangle(context, triangle, std::move(parent));
  const double effective_budget =
      std::max(error_budget, numerical_error_floor);
  if (assessment.error <= effective_budget)
    return {std::move(assessment.child_sum), assessment.error};
  if (triangle.depth >= context.options.maximum_subdivision_depth) {
    // Deterministic hard cap: keep the more accurate child sum and expose the
    // unresolved parent/child difference through the source integration audit.
    return {std::move(assessment.child_sum), assessment.error};
  }

  ShapeFactorIntegral result;
  const double child_weight_sum = row_total(assessment.child_sum);
  for (std::size_t child = 0; child < assessment.children.size(); ++child) {
    const double fraction =
        child_weight_sum > 0.0
            ? row_total(assessment.child_evaluations[child]) /
                  child_weight_sum
            : 0.25;
    auto refined = integrate_shape_factor_triangle(
        context, assessment.children[child], error_budget * fraction,
        numerical_error_floor,
        std::move(assessment.child_evaluations[child]),
        assessment.child_samples[child]);
    add_row(result.row, refined.row);
    result.estimated_l1_error += refined.estimated_l1_error;
  }
  return result;
}

double trace_shape_factor_point(
    const Scene& scene, const Geometry& geometry,
    const std::unordered_map<std::uint64_t, std::uint32_t>&
        primitive_to_state,
    const std::unordered_map<std::int32_t, std::uint32_t>&
        channel_to_column,
    const CustomRuntimeMap& custom_runtimes,
    const StructuredAnalyticIntersector* structured_intersector,
    const ShapeFactorFeatureEdges& feature_edges,
    const ShapeFactorDomainCandidates& domain_candidates,
    const SourcePoint& source, const ShapeFactorOptions& options,
    std::size_t transition_count, std::size_t detection_count,
    std::size_t loss_count, DenseRowAccumulator& output) {
  if (std::abs(source.stokes.q) > scene.numerics.energy_tolerance ||
      std::abs(source.stokes.u) > scene.numerics.energy_tolerance ||
      std::abs(source.stokes.v) > scene.numerics.energy_tolerance)
    throw std::runtime_error(
        "isotropic shape-factor sources currently require unpolarized "
        "Stokes input");
  if (scene.media.find(source.domain) == scene.media.end())
    throw std::runtime_error("shape-factor source domain is not declared");

  ShapeFactorTraceContext context{
      scene, geometry, primitive_to_state, channel_to_column,
      custom_runtimes, structured_intersector, source, options};
  std::vector<std::uint32_t> boundary_primitives;
  boundary_primitives.reserve(domain_candidates.triangles.size());
  std::vector<std::uint32_t> structured_integrals;
  for (const auto& candidate : domain_candidates.triangles) {
    if (dot(subtract(candidate.center, source.position),
            candidate.outward) <= 0.0)
      continue;
    const auto missing = std::numeric_limits<std::uint32_t>::max();
    const auto replacement =
        scene.mesh.triangle_source_analytic_primitive.empty()
            ? missing
            : scene.mesh.triangle_source_analytic_primitive.at(
                  candidate.index);
    if (options.backend != ShapeFactorBackend::generic_bvh &&
        replacement != missing &&
        scene.mesh.analytic_primitives.at(replacement).source_integral !=
            AnalyticSourceIntegral::none) {
      if (std::find(structured_integrals.begin(),
                    structured_integrals.end(), replacement) ==
          structured_integrals.end())
        structured_integrals.push_back(replacement);
      continue;
    }
    boundary_primitives.push_back(candidate.index);
  }
  std::vector<std::uint32_t> analytic_elements;
  analytic_elements.reserve(domain_candidates.analytic_elements.size());
  for (const auto& candidate : domain_candidates.analytic_elements) {
    if (dot(subtract(candidate.center, source.position),
            candidate.outward) > 0.0)
      analytic_elements.push_back(candidate.index);
  }
  const std::size_t boundary_count =
      boundary_primitives.size() + analytic_elements.size() +
      structured_integrals.size();
  if (boundary_count == 0)
    throw std::runtime_error(
        "shape-factor source has no outward-facing domain boundary");
  const double global_error_budget =
      options.relative_tolerance * source.stokes.i;
  const double primitive_error_budget =
      global_error_budget / boundary_count;
  const double primitive_numerical_error_floor =
      scene.numerics.energy_tolerance * source.stokes.i /
      boundary_count;

  DenseRowAccumulator dense_result;
  dense_result.reset(transition_count, detection_count, loss_count);
  double estimated_l1_error = 0.0;
  std::exception_ptr error;
#pragma omp parallel
  {
    thread_local DenseRowAccumulator local;
    local.reset(transition_count, detection_count, loss_count);
    double local_estimated_l1_error = 0.0;
    std::exception_ptr local_error;
#pragma omp for schedule(dynamic, 8)
    for (std::int64_t boundary_index = 0;
         boundary_index <
         static_cast<std::int64_t>(boundary_count);
         ++boundary_index) {
      if (local_error) continue;
      try {
        if (boundary_index >=
            static_cast<std::int64_t>(boundary_primitives.size() +
                                      analytic_elements.size())) {
          const auto structured_index =
              structured_integrals[boundary_index -
                                   boundary_primitives.size() -
                                   analytic_elements.size()];
          local_estimated_l1_error += accumulate_structured_disk(
              context, structured_index, transition_count, detection_count,
              loss_count, local);
          continue;
        }
        if (boundary_index >=
            static_cast<std::int64_t>(boundary_primitives.size())) {
          const auto analytic_index =
              analytic_elements[boundary_index -
                                boundary_primitives.size()];
          accumulate_shape_factor_analytic_element(
              context, analytic_index, local);
          continue;
        }
        const auto primitive = boundary_primitives[boundary_index];
        const auto indices = scene.mesh.triangles.at(primitive);
        ShapeFactorTriangle triangle{
            scene.mesh.vertices.at(indices[0]),
            scene.mesh.vertices.at(indices[1]),
            scene.mesh.vertices.at(indices[2]), primitive, 0,
            feature_edges.at(primitive)};
        auto integrated = integrate_shape_factor_triangle(
            context, triangle, primitive_error_budget,
            primitive_numerical_error_floor);
        local.add(integrated.row);
        local_estimated_l1_error += integrated.estimated_l1_error;
      } catch (...) {
        local_error = std::current_exception();
      }
    }
#pragma omp critical(oos_shape_factor_merge)
    {
      dense_result.add(local);
      estimated_l1_error += local_estimated_l1_error;
      if (local_error && !error) error = local_error;
    }
  }
  if (error) std::rethrow_exception(error);
  const double accounted = dense_result.total();
  if (!(accounted > 0.0) || !std::isfinite(accounted))
    throw std::runtime_error(
        "shape-factor source produced no finite boundary weight");
  // A hard cap can stop while a non-convex boundary triangle is only partly
  // visible. Its centroid decision then temporarily under- or over-counts a
  // small solid-angle region. Deterministic source quadratures are normalized
  // by contract; include the normalization correction in the L1 error audit.
  estimated_l1_error += std::abs(accounted - source.stokes.i);
  dense_result.scale_values(source.stokes.i / accounted);
  output.add(dense_result);
  return estimated_l1_error;
}

CsrMatrix maps_to_csr(
    const std::vector<std::map<std::uint32_t, double>>& rows,
    std::uint64_t columns) {
  CsrMatrix matrix;
  matrix.rows = rows.size();
  matrix.cols = columns;
  matrix.indptr.push_back(0);
  for (const auto& row : rows) {
    for (const auto& [column, value] : row) {
      if (value == 0.0) continue;
      matrix.indices.push_back(column);
      matrix.data.push_back(value);
    }
    matrix.indptr.push_back(matrix.data.size());
  }
  return matrix;
}

CsrMatrix payload_csr(const PluginBuildResult& payload,
                      const std::string& root) {
  const auto& shape = payload.u64.at(root + "/shape").values;
  const auto& indptr = payload.u64.at(root + "/indptr").values;
  const auto& index_values = payload.u64.at(root + "/indices").values;
  CsrMatrix result;
  if (shape.size() != 2) throw std::runtime_error("plugin CSR shape is invalid");
  result.rows = shape[0];
  result.cols = shape[1];
  result.indptr = indptr;
  result.indices.reserve(index_values.size());
  for (const auto value : index_values) {
    if (value > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("plugin CSR index exceeds uint32");
    result.indices.push_back(static_cast<std::uint32_t>(value));
  }
  result.data = payload.f64.at(root + "/data").values;
  return result;
}

void apply_provenance(OperatorSet& result, const Scene& scene) {
  result.energy_eV = scene.energy_eV;
  result.scene_sha256 =
      scene.source_path.empty() ? "in-memory" : sha256_file(scene.source_path);
  result.geometry_sha256 = scene.geometry_path.empty()
                               ? "in-memory"
                               : sha256_file(scene.geometry_path);
  result.surface_basis_sha256 =
      scene.surface_basis_path.empty()
          ? "embedded"
          : sha256_file(scene.surface_basis_path);
#ifdef OOS_DEPS_LOCK_SHA256
  result.dependency_lock_sha256 = OOS_DEPS_LOCK_SHA256;
#endif
#ifdef OOS_GIT_COMMIT
  result.code_commit = OOS_GIT_COMMIT;
#endif
  std::string cache_material =
      result.scene_sha256 + result.geometry_sha256 +
      result.surface_basis_sha256 +
      result.dependency_lock_sha256 + result.code_commit +
      std::to_string(result.energy_eV);
  for (const auto& [id, surface] : scene.surfaces) {
    (void)id;
    if (surface.kind == SurfaceKind::custom_local ||
        surface.kind == SurfaceKind::custom_nonlocal) {
      cache_material += sha256_file(surface.plugin_path);
      if (!surface.plugin_config_json.empty()) {
        const auto config =
            nlohmann::json::parse(surface.plugin_config_json);
        for (const auto* key :
             {"precomputed_operator_hdf5", "precomputed_block_hdf5",
              "factorized_block_hdf5"}) {
          if (!config.contains(key)) continue;
          const auto path =
              std::filesystem::path(config.at(key).get<std::string>());
          if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error(
                std::string("custom operator input is not a regular file: ") +
                path.string());
          cache_material += key;
          cache_material += sha256_file(path);
        }
      }
    }
  }
  result.cache_key_sha256 = sha256_string(cache_material);
}

std::vector<std::string> payload_loss_names(
    const PluginBuildResult& payload, std::uint64_t count) {
  const auto metadata = nlohmann::json::parse(payload.metadata_json);
  if (!metadata.contains("loss_names"))
    throw std::runtime_error("nonlocal plugin metadata lacks loss_names");
  const auto names =
      metadata.at("loss_names").get<std::vector<std::string>>();
  if (names.size() != count)
    throw std::runtime_error("nonlocal plugin loss_names shape mismatch");
  return names;
}

struct NonlocalEgress {
  std::uint64_t surface_element{};
  std::array<double, 3> barycentric{};
  std::uint64_t side{};
  Vec3 direction_local{};
  Stokes stokes{1.0, 0.0, 0.0, 0.0};
  Vec3 reference_axis_local{1.0, 0.0, 0.0};
};

std::vector<NonlocalEgress> payload_egress(
    const PluginBuildResult& payload) {
  const auto& elements =
      payload.u64.at("/nonlocal/egress/surface_element");
  const auto& barycentric =
      payload.f64.at("/nonlocal/egress/barycentric");
  const auto& sides = payload.u64.at("/nonlocal/egress/side");
  const auto& directions =
      payload.f64.at("/nonlocal/egress/direction_local");
  const auto stokes =
      payload.f64.find("/nonlocal/egress/stokes");
  const auto reference_axes =
      payload.f64.find("/nonlocal/egress/reference_axis_local");
  if (elements.shape.size() != 1 || sides.shape != elements.shape ||
      barycentric.shape !=
          std::vector<std::uint64_t>{elements.shape[0], 3} ||
      directions.shape !=
          std::vector<std::uint64_t>{elements.shape[0], 3} ||
      (stokes != payload.f64.end() &&
       stokes->second.shape !=
           std::vector<std::uint64_t>{elements.shape[0], 4}) ||
      (reference_axes != payload.f64.end() &&
       reference_axes->second.shape !=
           std::vector<std::uint64_t>{elements.shape[0], 3}))
    throw std::runtime_error("nonlocal egress dataset shapes disagree");
  std::vector<NonlocalEgress> result(elements.shape[0]);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i].surface_element = elements.values.at(i);
    result[i].barycentric = {barycentric.values.at(3 * i),
                             barycentric.values.at(3 * i + 1),
                             barycentric.values.at(3 * i + 2)};
    result[i].side = sides.values.at(i);
    result[i].direction_local = {directions.values.at(3 * i),
                                 directions.values.at(3 * i + 1),
                                 directions.values.at(3 * i + 2)};
    if (stokes != payload.f64.end())
      result[i].stokes = {stokes->second.values.at(4 * i),
                          stokes->second.values.at(4 * i + 1),
                          stokes->second.values.at(4 * i + 2),
                          stokes->second.values.at(4 * i + 3)};
    if (reference_axes != payload.f64.end())
      result[i].reference_axis_local = {
          reference_axes->second.values.at(3 * i),
          reference_axes->second.values.at(3 * i + 1),
          reference_axes->second.values.at(3 * i + 2)};
    const double barycentric_sum =
        std::accumulate(result[i].barycentric.begin(),
                        result[i].barycentric.end(), 0.0);
    for (const auto value : result[i].barycentric)
      if (!std::isfinite(value) || value < 0.0)
        throw std::runtime_error(
            "nonlocal egress barycentric coordinate is invalid");
    if (std::abs(barycentric_sum - 1.0) > 1.0e-12)
      throw std::runtime_error(
          "nonlocal egress barycentric coordinates do not close");
    if (result[i].side > 1)
      throw std::runtime_error("nonlocal egress side must be zero or one");
    if (!std::isfinite(result[i].direction_local.x) ||
        !std::isfinite(result[i].direction_local.y) ||
        !std::isfinite(result[i].direction_local.z) ||
        result[i].direction_local.z <= 0.0)
      throw std::runtime_error(
          "nonlocal egress direction must point into its declared side");
    result[i].direction_local = normalized(result[i].direction_local);
    const auto& polarization = result[i].stokes;
    if (!std::isfinite(polarization.i) ||
        !std::isfinite(polarization.q) ||
        !std::isfinite(polarization.u) ||
        !std::isfinite(polarization.v) ||
        std::abs(polarization.i - 1.0) > 1.0e-12 ||
        polarization.q * polarization.q +
                polarization.u * polarization.u +
                polarization.v * polarization.v >
            1.0 + 1.0e-12)
      throw std::runtime_error(
          "nonlocal egress Stokes vector must have unit intensity and "
          "physical polarization");
    if (!std::isfinite(result[i].reference_axis_local.x) ||
        !std::isfinite(result[i].reference_axis_local.y) ||
        !std::isfinite(result[i].reference_axis_local.z))
      throw std::runtime_error(
          "nonlocal egress reference axis is invalid");
  }
  return result;
}

std::vector<std::uint32_t> surface_primitives(const Scene& scene,
                                              std::uint32_t surface_id) {
  std::vector<std::uint32_t> result;
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.surface_id.size(); ++primitive)
    if (scene.mesh.surface_id[primitive] == surface_id &&
        (scene.mesh.triangle_transport.empty() ||
         scene.mesh.triangle_transport.at(primitive) != 0))
      result.push_back(primitive);
  return result;
}

WeightedRay egress_ray(const Scene& scene, std::uint32_t surface_id,
                       const NonlocalEgress& egress, double nudge,
                       double weight = 1.0) {
  const AnalyticSurfaceElement* analytic_element = nullptr;
  const AnalyticPrimitive* analytic_primitive = nullptr;
  for (const auto& element : scene.mesh.analytic_surface_elements) {
    const auto& primitive =
        scene.mesh.analytic_primitives.at(element.primitive_index);
    if (primitive.surface_id != surface_id ||
        element.surface_element != egress.surface_element)
      continue;
    if (analytic_element != nullptr)
      throw std::runtime_error(
          "nonlocal egress surface element is ambiguous");
    analytic_element = &element;
    analytic_primitive = &primitive;
  }
  if (analytic_element != nullptr) {
    const Vec3 point = analytic_element->center_mm;
    const Vec3 geometric_normal = normalized(analytic_element->normal);
    const Vec3 side_normal =
        egress.side == 1 ? geometric_normal
                         : scale(geometric_normal, -1.0);
    Vec3 tangent = analytic_primitive->axis_x;
    if (analytic_element->coordinates ==
        AnalyticSurfaceCoordinates::cylinder_phi_z) {
      const double phi =
          0.5 * (analytic_element->bounds[0] +
                 analytic_element->bounds[1]);
      tangent = normalized(
          add(scale(analytic_primitive->axis_x, -std::sin(phi)),
              scale(analytic_primitive->axis_y, std::cos(phi))));
    }
    tangent = normalized(
        subtract(tangent, scale(side_normal, dot(tangent, side_normal))));
    const Vec3 bitangent = cross(side_normal, tangent);
    const Vec3 direction = normalized(
        add(scale(tangent, egress.direction_local.x),
            add(scale(bitangent, egress.direction_local.y),
                scale(side_normal, egress.direction_local.z))));
    Vec3 reference_axis =
        add(scale(tangent, egress.reference_axis_local.x),
            add(scale(bitangent, egress.reference_axis_local.y),
                scale(side_normal, egress.reference_axis_local.z)));
    reference_axis =
        subtract(reference_axis,
                 scale(direction, dot(reference_axis, direction)));
    if (norm(reference_axis) < 1.0e-14)
      reference_axis =
          subtract(tangent, scale(direction, dot(tangent, direction)));
    if (norm(reference_axis) < 1.0e-14)
      reference_axis =
          subtract(bitangent, scale(direction, dot(bitangent, direction)));
    reference_axis = normalized(reference_axis);
    const auto domain =
        egress.side == 1 ? analytic_primitive->plus_domain_id
                         : analytic_primitive->minus_domain_id;
    if (domain < 0)
      throw std::runtime_error(
          "nonlocal analytic egress points outside all declared media");
    return {{add(point, scale(side_normal, nudge)), direction},
            {weight, weight * egress.stokes.q, weight * egress.stokes.u,
             weight * egress.stokes.v},
            reference_axis, domain, 0, std::nullopt};
  }
  const auto primitives = surface_primitives(scene, surface_id);
  if (egress.surface_element >= primitives.size())
    throw std::runtime_error(
        "nonlocal egress surface element lies outside its own surface");
  const auto primitive = primitives.at(egress.surface_element);
  const auto triangle = scene.mesh.triangles.at(primitive);
  Vec3 point{};
  for (int vertex = 0; vertex < 3; ++vertex)
    point = add(point,
                scale(scene.mesh.vertices.at(triangle[vertex]),
                      egress.barycentric[vertex]));
  const Vec3 geometric_normal = triangle_normal(scene, primitive);
  const Vec3 side_normal =
      egress.side == 1 ? geometric_normal : scale(geometric_normal, -1.0);
  const Vec3 tangent = normalized(
      subtract(scene.mesh.vertices.at(triangle[1]),
               scene.mesh.vertices.at(triangle[0])));
  const Vec3 bitangent = cross(side_normal, tangent);
  const Vec3 direction = normalized(
      add(scale(tangent, egress.direction_local.x),
          add(scale(bitangent, egress.direction_local.y),
              scale(side_normal, egress.direction_local.z))));
  Vec3 reference_axis =
      add(scale(tangent, egress.reference_axis_local.x),
          add(scale(bitangent, egress.reference_axis_local.y),
              scale(side_normal, egress.reference_axis_local.z)));
  reference_axis =
      subtract(reference_axis,
               scale(direction, dot(reference_axis, direction)));
  if (norm(reference_axis) < 1.0e-14)
    reference_axis =
        subtract(tangent, scale(direction, dot(tangent, direction)));
  if (norm(reference_axis) < 1.0e-14)
    reference_axis =
        subtract(bitangent, scale(direction, dot(bitangent, direction)));
  reference_axis = normalized(reference_axis);
  const auto domain =
      egress.side == 1 ? scene.mesh.plus_domain_id.at(primitive)
                       : scene.mesh.minus_domain_id.at(primitive);
  if (domain < 0)
    throw std::runtime_error(
        "nonlocal egress ray points outside all declared media");
  return {{add(point, scale(side_normal, nudge)),
           direction},
          {weight, weight * egress.stokes.q, weight * egress.stokes.u,
           weight * egress.stokes.v},
          reference_axis, domain, 0, std::nullopt};
}

void validate_intrinsic_nonlocal_payload(const PluginBuildResult& payload,
                                         double tolerance) {
  const std::array<std::string, 5> forbidden{
      "/nonlocal/detection", "/nonlocal/channel_id",
      "/nonlocal/to_local", "/nonlocal/to_local_primitive_id", "/operators"};
  const auto reject_forbidden = [&forbidden](const auto& datasets) {
    for (const auto& [path, dataset] : datasets) {
      (void)dataset;
      for (const auto& prefix : forbidden)
        if (path == prefix || path.rfind(prefix + "/", 0) == 0)
          throw std::runtime_error(
              "nonlocal surface payload contains geometry-coupled field " +
              path);
    }
  };
  reject_forbidden(payload.f64);
  reject_forbidden(payload.u64);
  const auto metadata = nlohmann::json::parse(payload.metadata_json);
  if (metadata.value("execution", std::string{}) == "function") {
    const auto egress = payload_egress(payload);
    if (!metadata.contains("state_count") ||
        !metadata.contains("loss_names") ||
        metadata.at("state_count").get<std::uint64_t>() == 0 ||
        egress.empty())
      throw std::runtime_error(
          "functional nonlocal payload descriptor is incomplete");
    return;
  }
  const auto internal =
      payload_csr(payload, "/nonlocal/internal_transition");
  const auto emission = payload_csr(payload, "/nonlocal/emission");
  const auto losses = payload_csr(payload, "/nonlocal/internal_losses");
  const auto egress = payload_egress(payload);
  internal.validate();
  emission.validate();
  losses.validate();
  if (internal.rows != internal.cols || emission.rows != internal.rows ||
      losses.rows != internal.rows || emission.cols != egress.size())
    throw std::runtime_error(
        "nonlocal intrinsic block dimensions do not agree");
  for (std::uint64_t row = 0; row < internal.rows; ++row) {
    double total = 0.0;
    const auto accumulate_row = [row, &total](const CsrMatrix& matrix) {
      for (auto entry = matrix.indptr[row];
           entry < matrix.indptr[row + 1]; ++entry) {
        if (!std::isfinite(matrix.data[entry]) || matrix.data[entry] < 0.0)
          throw std::runtime_error(
              "nonlocal intrinsic block contains invalid weight");
        total += matrix.data[entry];
      }
    };
    accumulate_row(internal);
    accumulate_row(emission);
    accumulate_row(losses);
    if (std::abs(total - 1.0) > tolerance)
      throw std::runtime_error(
          "nonlocal intrinsic state row does not conserve probability");
  }
}

CustomRuntimeMap create_custom_runtimes(const Scene& scene,
                                        std::uint64_t local_state_count) {
  std::vector<std::uint32_t> surface_ids;
  for (const auto& [id, surface] : scene.surfaces)
    if (surface.kind == SurfaceKind::custom_local ||
        surface.kind == SurfaceKind::custom_nonlocal)
      surface_ids.push_back(id);
  std::sort(surface_ids.begin(), surface_ids.end());
  CustomRuntimeMap result;
  std::uint64_t offset = local_state_count;
  for (const auto id : surface_ids) {
    const auto& surface = scene.surfaces.at(id);
    CustomSurfaceRuntime runtime;
    runtime.plugin = std::make_shared<SurfacePlugin>(surface.plugin_path);
    runtime.config_json = surface.plugin_config_json;
    const auto expected =
        surface.kind == SurfaceKind::custom_local ? PluginLocality::local
                                                  : PluginLocality::nonlocal;
    if (runtime.plugin->locality() != expected)
      throw std::runtime_error("custom surface/plugin locality mismatch");
    if (expected == PluginLocality::nonlocal) {
      runtime.payload =
          runtime.plugin->build(runtime.config_json, scene.energy_eV);
      validate_intrinsic_nonlocal_payload(runtime.payload,
                                          scene.numerics.energy_tolerance);
      const auto config = nlohmann::json::parse(runtime.config_json);
      if (!config.contains("nonlocal_domain_id"))
        throw std::runtime_error(
            "nonlocal custom surface requires nonlocal_domain_id");
      runtime.nonlocal_domain_id =
          config.at("nonlocal_domain_id").get<std::int32_t>();
      if (!scene.media.count(runtime.nonlocal_domain_id))
        throw std::runtime_error(
            "nonlocal custom surface domain is not a declared medium");
      const auto metadata =
          nlohmann::json::parse(runtime.payload.metadata_json);
      runtime.functional =
          metadata.value("execution", std::string{}) == "function";
      if (runtime.functional) {
        FunctionOperator function(surface.plugin_path,
                                  runtime.config_json, scene.energy_eV);
        validate_function_operator(function,
                                   scene.numerics.energy_tolerance);
        runtime.function_descriptor = function.descriptor();
        runtime.state_count =
            runtime.function_descriptor.input_state_count;
        if (metadata.at("state_count").get<std::uint64_t>() !=
                runtime.state_count ||
            payload_egress(runtime.payload).size() !=
                runtime.function_descriptor.egress_count)
          throw std::runtime_error(
              "functional nonlocal payload and runtime dimensions disagree");
      } else {
        const auto transition =
            payload_csr(runtime.payload, "/nonlocal/internal_transition");
        if (transition.rows != transition.cols)
          throw std::runtime_error(
              "nonlocal plugin transition block must be square");
        runtime.state_count = transition.rows;
      }
      runtime.state_offset = offset;
      offset += runtime.state_count;
    }
    result.emplace(id, std::move(runtime));
  }
  return result;
}

}  // namespace

struct SourceTraceRuntime::Impl {
  const Scene& scene;
  const OperatorSet& operators;
  Geometry geometry;
  StructuredAnalyticIntersector structured_geometry;
  LocalSurfaceBasis local_basis;
  CustomRuntimeMap custom_runtimes;
  std::unordered_map<std::int32_t, std::uint32_t> channel_to_column;
  std::unordered_map<std::int32_t, ShapeFactorDomainCandidates>
      domain_candidates;
  mutable std::map<double, ShapeFactorFeatureEdges> feature_edge_cache;
  mutable std::mutex feature_edge_mutex;

  Impl(const Scene& scene_value, const OperatorSet& operator_value)
      : scene(scene_value),
        operators(operator_value),
        geometry(scene_value),
        structured_geometry(scene_value, geometry),
        local_basis(make_local_surface_basis(scene_value)),
        custom_runtimes(create_custom_runtimes(
            scene_value,
            static_cast<std::uint32_t>(local_basis.state_emitters.size()))) {
    if (operators.ray_origin_offset_mm > 0.0 &&
        std::abs(geometry.ray_origin_offset_mm() -
                 operators.ray_origin_offset_mm) >
            scene.numerics.energy_tolerance)
      throw std::runtime_error(
          "scene ray-origin offset does not match the operator cache");

    std::uint64_t expected_states = local_basis.state_emitters.size();
    for (const auto& [surface_id, runtime] : custom_runtimes) {
      (void)surface_id;
      expected_states = std::max(
          expected_states, runtime.state_offset + runtime.state_count);
    }
    if (expected_states != operators.transition.rows)
      throw std::runtime_error("scene states do not match operator cache");
    for (std::uint32_t column = 0;
         column < operators.channel_ids.size(); ++column)
      channel_to_column[operators.channel_ids[column]] = column;

    // Build domain-adjacent source-boundary structure-of-arrays once. The
    // source-dependent facing test below then touches only compact candidates
    // instead of rescanning and rebuilding normals for the complete mesh.
    for (std::uint32_t primitive = 0;
         primitive < scene.mesh.triangles.size(); ++primitive) {
      const bool source_quadrature =
          scene.mesh.triangle_source_quadrature.empty()
              ? (scene.mesh.triangle_transport.empty() ||
                 scene.mesh.triangle_transport.at(primitive) != 0)
              : scene.mesh.triangle_source_quadrature.at(primitive) != 0;
      if (!source_quadrature) continue;
      const Vec3 center = triangle_center(scene, primitive);
      const Vec3 normal = triangle_normal(scene, primitive);
      const auto minus = scene.mesh.minus_domain_id.at(primitive);
      const auto plus = scene.mesh.plus_domain_id.at(primitive);
      if (scene.media.count(minus))
        domain_candidates[minus].triangles.push_back(
            {primitive, center, normal});
      if (plus != minus && scene.media.count(plus))
        domain_candidates[plus].triangles.push_back(
            {primitive, center, scale(normal, -1.0)});
    }
    for (std::uint32_t index = 0;
         index < scene.mesh.analytic_surface_elements.size(); ++index) {
      const auto& element =
          scene.mesh.analytic_surface_elements.at(index);
      if (!element.source_quadrature) continue;
      const auto& primitive =
          scene.mesh.analytic_primitives.at(element.primitive_index);
      if (scene.media.count(primitive.minus_domain_id))
        domain_candidates[primitive.minus_domain_id]
            .analytic_elements.push_back(
                {index, element.center_mm, element.normal});
      if (primitive.plus_domain_id != primitive.minus_domain_id &&
          scene.media.count(primitive.plus_domain_id))
        domain_candidates[primitive.plus_domain_id]
            .analytic_elements.push_back(
                {index, element.center_mm, element.normal});
    }
  }

  const ShapeFactorFeatureEdges& feature_edges(double dihedral) const {
    std::lock_guard<std::mutex> lock(feature_edge_mutex);
    auto found = feature_edge_cache.find(dihedral);
    if (found == feature_edge_cache.end())
      found = feature_edge_cache
                  .emplace(dihedral, build_shape_factor_feature_edges(
                                         scene, dihedral))
                  .first;
    return found->second;
  }
};

SourceTraceRuntime::SourceTraceRuntime(
    const Scene& scene, const OperatorSet& operators) {
  // Validation is intentionally once per runtime rather than once per
  // coordinate/batch call; for production operator sets this is a measurable
  // full sparse-matrix traversal.
  operators.validate();
  impl_ = std::make_unique<Impl>(scene, operators);
}

SourceTraceRuntime::~SourceTraceRuntime() = default;
SourceTraceRuntime::SourceTraceRuntime(SourceTraceRuntime&&) noexcept =
    default;
SourceTraceRuntime& SourceTraceRuntime::operator=(
    SourceTraceRuntime&&) noexcept = default;

std::string OperatorBuilder::cache_key(const Scene& scene) {
  OperatorSet provenance;
  apply_provenance(provenance, scene);
  return provenance.cache_key_sha256;
}

OperatorSet OperatorBuilder::build(const Scene& scene) {
  SceneValidator::validate(scene).throw_if_invalid();
  Geometry geometry(scene);
  const auto local_basis = make_local_surface_basis(scene);
  const auto& state_emitters = local_basis.state_emitters;
  const auto& primitive_to_state = local_basis.primitive_to_state;
  std::vector<std::int32_t> channels;
  for (std::uint32_t primitive = 0;
       primitive < scene.mesh.triangles.size(); ++primitive) {
    if (!scene.mesh.triangle_transport.empty() &&
        !scene.mesh.triangle_transport.at(primitive))
      continue;
    const auto& surface = scene.surfaces.at(scene.mesh.surface_id[primitive]);
    if (surface.kind == SurfaceKind::sensitive ||
        (surface.kind == SurfaceKind::custom_local &&
         scene.mesh.channel_id[primitive] >= 0))
      channels.push_back(scene.mesh.channel_id[primitive]);
  }
  for (const auto& primitive : scene.mesh.analytic_primitives) {
    const auto& surface = scene.surfaces.at(primitive.surface_id);
    if (surface.kind == SurfaceKind::sensitive ||
        (surface.kind == SurfaceKind::custom_local &&
         primitive.channel_id >= 0))
      channels.push_back(primitive.channel_id);
  }
  const auto custom_runtimes =
      create_custom_runtimes(scene, state_emitters.size());
  std::uint64_t total_states = state_emitters.size();
  std::vector<std::string> state_labels = local_basis.state_labels;
  std::vector<std::string> loss_names{
      "bulk_absorption", "surface_absorption", "escape",
      "specular_or_model_truncation", "float32_intersection_miss"};
  std::unordered_map<std::uint32_t, std::uint64_t> plugin_loss_offset;
  for (const auto& [surface_id, runtime] : custom_runtimes) {
    if (runtime.plugin->locality() != PluginLocality::nonlocal) continue;
    const auto egress = payload_egress(runtime.payload);
    std::uint64_t loss_count = 0;
    if (runtime.functional) {
      if (runtime.function_descriptor.egress_count != egress.size())
        throw std::runtime_error(
            "functional nonlocal egress dimensions disagree");
      loss_count = runtime.function_descriptor.loss_count;
    } else {
      const auto emission =
          payload_csr(runtime.payload, "/nonlocal/emission");
      const auto losses =
          payload_csr(runtime.payload, "/nonlocal/internal_losses");
      if (emission.rows != runtime.state_count ||
          emission.cols != egress.size() ||
          losses.rows != runtime.state_count)
        throw std::runtime_error(
            "nonlocal plugin block row dimensions disagree");
      loss_count = losses.cols;
    }
    plugin_loss_offset[surface_id] = loss_names.size();
    for (const auto& name : payload_loss_names(runtime.payload, loss_count))
      loss_names.push_back(scene.surfaces.at(surface_id).name + "/" + name);
    total_states =
        std::max(total_states, runtime.state_offset + runtime.state_count);
  }
  state_labels.resize(total_states);
  for (const auto& [surface_id, runtime] : custom_runtimes)
    for (std::uint64_t state = 0; state < runtime.state_count; ++state)
      state_labels.at(runtime.state_offset + state) =
          scene.surfaces.at(surface_id).name + "/state:" +
          std::to_string(state);
  std::sort(channels.begin(), channels.end());
  channels.erase(std::unique(channels.begin(), channels.end()), channels.end());
  std::unordered_map<std::int32_t, std::uint32_t> channel_to_column;
  for (std::uint32_t i = 0; i < channels.size(); ++i)
    channel_to_column[channels[i]] = i;
  if (total_states == 0)
    throw std::runtime_error("scene has no transport states");
  if (channels.empty())
    throw std::runtime_error("scene has no sensitive channels");

  std::vector<std::map<std::uint32_t, double>> transition_rows(
      total_states);
  std::vector<std::map<std::uint32_t, double>> detection_rows(
      total_states);
  std::vector<std::map<std::uint32_t, double>> loss_rows(
      total_states);
  std::vector<FunctionBlock> function_blocks;
  const auto mu2_order = scene.numerics.lambertian_mu2_order;
  const auto phi_count = scene.numerics.lambertian_phi_count;
  const auto lambertian_mu2 = gauss_legendre_unit_interval(mu2_order);
  std::exception_ptr local_state_error;
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t state = 0;
       state < static_cast<std::int64_t>(state_emitters.size()); ++state) {
    try {
      RowAccumulation row;
      double state_area = 0.0;
      for (const auto& emitter : state_emitters[state])
        state_area += emitter.area_mm2;
      if (!(state_area > 0.0))
        throw std::runtime_error("surface basis has zero area");
      for (const auto& emitter : state_emitters[state]) {
        const double area_fraction = emitter.area_mm2 / state_area;
        const Vec3 geometric_normal = emitter.normal;
        const bool primary_is_minus =
            emitter.minus_domain_id == scene.primary_domain;
        const Vec3 inward =
            primary_is_minus ? scale(geometric_normal, -1.0)
                             : geometric_normal;
        const Vec3 tangent = emitter.tangent;
        const Vec3 bitangent = cross(inward, tangent);
        const Vec3 center = emitter.center;
        for (std::uint32_t radial = 0; radial < mu2_order; ++radial) {
          const double mu = std::sqrt(lambertian_mu2[radial].first);
          const double angular_weight =
              lambertian_mu2[radial].second / phi_count;
          const double transverse = std::sqrt(1.0 - mu * mu);
          for (std::uint32_t angular = 0; angular < phi_count; ++angular) {
            const double phi =
                2.0 * pi * (angular + 0.5) / phi_count;
            const Vec3 direction =
                normalized(add(
                    scale(inward, mu),
                    add(scale(tangent, transverse * std::cos(phi)),
                        scale(bitangent, transverse * std::sin(phi)))));
            trace_branch(
                scene, geometry, primitive_to_state, channel_to_column,
                custom_runtimes,
                {{add(center,
                      scale(inward, geometry.ray_origin_offset_mm())),
                  direction},
                 {area_fraction * angular_weight, 0, 0, 0}, tangent,
                 scene.primary_domain, 0, std::nullopt},
                row);
          }
        }
      }
      transition_rows[state] = ordered_map(row.transition);
      detection_rows[state] = ordered_map(row.detection);
      loss_rows[state] = ordered_map(row.losses);
    } catch (...) {
#pragma omp critical(oos_local_state_build_error)
      {
        if (!local_state_error)
          local_state_error = std::current_exception();
      }
    }
  }
  if (local_state_error) std::rethrow_exception(local_state_error);
  for (const auto& runtime_entry : custom_runtimes) {
    const auto surface_id = runtime_entry.first;
    const auto& runtime = runtime_entry.second;
    if (runtime.plugin->locality() != PluginLocality::nonlocal) continue;
    const auto egress = payload_egress(runtime.payload);
    std::vector<RowAccumulation> egress_rows(egress.size());
    std::exception_ptr egress_error;
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(egress.size()); ++index) {
      try {
        trace_branch(scene, geometry, primitive_to_state, channel_to_column,
                     custom_runtimes,
                     egress_ray(scene, surface_id, egress[index],
                                geometry.ray_origin_offset_mm()),
                     egress_rows[index]);
      } catch (...) {
#pragma omp critical(oos_egress_build_error)
        {
          if (!egress_error) egress_error = std::current_exception();
        }
      }
    }
    if (egress_error) std::rethrow_exception(egress_error);
    if (runtime.functional) {
      std::vector<std::map<std::uint32_t, double>> to_transition(
          egress.size());
      std::vector<std::map<std::uint32_t, double>> to_detection(
          egress.size());
      std::vector<std::map<std::uint32_t, double>> to_losses(
          egress.size());
      for (std::size_t index = 0; index < egress.size(); ++index) {
        to_transition[index] = ordered_map(egress_rows[index].transition);
        to_detection[index] = ordered_map(egress_rows[index].detection);
        to_losses[index] = ordered_map(egress_rows[index].losses);
      }
      FunctionBlock block;
      block.name = scene.surfaces.at(surface_id).name;
      block.library_path = scene.surfaces.at(surface_id).plugin_path;
      block.config_json = runtime.config_json;
      block.state_offset = runtime.state_offset;
      block.state_count = runtime.state_count;
      block.egress_count = egress.size();
      block.intrinsic_loss_count =
          runtime.function_descriptor.loss_count;
      block.contraction_bound =
          runtime.function_descriptor.contraction_bound;
      block.egress_to_transition =
          maps_to_csr(to_transition, total_states);
      block.egress_to_detection =
          maps_to_csr(to_detection, channels.size());
      block.egress_to_losses =
          maps_to_csr(to_losses, loss_names.size());
      for (std::uint64_t loss = 0;
           loss < block.intrinsic_loss_count; ++loss)
        block.intrinsic_loss_columns.push_back(
            static_cast<std::uint32_t>(
                plugin_loss_offset.at(surface_id) + loss));
      function_blocks.push_back(std::move(block));
      continue;
    }
    const auto internal =
        payload_csr(runtime.payload, "/nonlocal/internal_transition");
    const auto emission =
        payload_csr(runtime.payload, "/nonlocal/emission");
    const auto losses =
        payload_csr(runtime.payload, "/nonlocal/internal_losses");
    for (std::uint64_t row = 0; row < runtime.state_count; ++row) {
      const auto global_row =
          static_cast<std::uint32_t>(runtime.state_offset + row);
      for (std::uint64_t entry = internal.indptr[row];
           entry < internal.indptr[row + 1]; ++entry)
        transition_rows[global_row][static_cast<std::uint32_t>(
            runtime.state_offset + internal.indices[entry])] +=
            internal.data[entry];
      for (std::uint64_t entry = emission.indptr[row];
           entry < emission.indptr[row + 1]; ++entry) {
        const auto& traced = egress_rows.at(emission.indices[entry]);
        const double factor = emission.data[entry];
        for (const auto& [column, value] : traced.transition)
          transition_rows[global_row][column] += factor * value;
        for (const auto& [column, value] : traced.detection)
          detection_rows[global_row][column] += factor * value;
        for (const auto& [column, value] : traced.losses)
          loss_rows[global_row][column] += factor * value;
      }
      for (std::uint64_t entry = losses.indptr[row];
           entry < losses.indptr[row + 1]; ++entry)
        loss_rows[global_row][static_cast<std::uint32_t>(
            plugin_loss_offset.at(surface_id) + losses.indices[entry])] +=
            losses.data[entry];
    }
  }
  OperatorSet result;
  result.transition = maps_to_csr(transition_rows, total_states);
  result.detection = maps_to_csr(detection_rows, channels.size());
  result.losses = maps_to_csr(loss_rows, loss_names.size());
  result.function_blocks = std::move(function_blocks);
  result.state_labels = std::move(state_labels);
  result.channel_ids = std::move(channels);
  result.loss_names = std::move(loss_names);
  result.tolerance = scene.numerics.neumann_tolerance;
  result.maximum_iterations = scene.numerics.maximum_diffuse_bounces;
  result.ray_origin_offset_mm = geometry.ray_origin_offset_mm();
  apply_provenance(result, scene);
  result.validate();
  return result;
}

SourceBatch SourceTraceRuntime::trace(
    const std::vector<SourceQuadrature>& quadratures) const {
  const auto& scene = impl_->scene;
  const auto& operators = impl_->operators;
  const auto& geometry = impl_->geometry;
  const auto& primitive_to_state =
      impl_->local_basis.primitive_to_state;
  const auto& custom_runtimes = impl_->custom_runtimes;
  const auto& channel_to_column = impl_->channel_to_column;

  SourceBatch result;
  result.count = quadratures.size();
  result.initial_states.assign(result.count * operators.transition.rows, 0.0);
  result.direct_detection.assign(result.count * operators.detection.cols, 0.0);
  result.direct_losses.assign(result.count * operators.losses.cols, 0.0);
  result.source_integration_l1_error_estimate.assign(result.count, 0.0);
  const auto store_row = [&](std::size_t index,
                             const RowAccumulation& row) {
    for (const auto& [column, value] : row.transition)
      result.initial_states[index * operators.transition.rows + column] =
          value;
    for (const auto& [column, value] : row.detection)
      result.direct_detection[index * operators.detection.cols + column] =
          value;
    for (const auto& [loss, value] : row.losses) {
      if (loss >= operators.losses.cols)
        throw std::runtime_error("source trace produced an unknown loss");
      result.direct_losses[index * operators.losses.cols + loss] = value;
    }
  };

  // Ordinary ray quadratures parallelize naturally over batched sources.
  // Shape-factor sources instead parallelize over their hundreds of
  // thousands of boundary triangles, so they must not be enclosed by this
  // outer OpenMP region (nested OpenMP is disabled on production workers).
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(quadratures.size()); ++index) {
    if (quadratures[index].integration ==
        SourceIntegration::isotropic_surface_shape_factor)
      continue;
    RowAccumulation row;
    std::vector<Ray> root_rays;
    std::vector<std::int32_t> root_domains;
    root_rays.reserve(quadratures[index].rays.size());
    root_domains.reserve(quadratures[index].rays.size());
    for (const auto& source_ray : quadratures[index].rays) {
      root_rays.push_back(source_ray.ray);
      root_domains.push_back(source_ray.domain);
    }
    const auto root_hits =
        geometry.intersect_batch(root_rays, root_domains);
    for (std::size_t ray_index = 0;
         ray_index < quadratures[index].rays.size(); ++ray_index) {
      const auto& source_ray = quadratures[index].rays[ray_index];
      trace_branch(
          scene, geometry, primitive_to_state, channel_to_column,
          custom_runtimes,
          {source_ray.ray, source_ray.stokes, source_ray.reference_axis,
           source_ray.domain, 0, root_hits[ray_index]},
          row);
    }
    store_row(static_cast<std::size_t>(index), row);
  }

  std::vector<std::size_t> shape_factor_indices;
  std::map<double, const ShapeFactorFeatureEdges*> feature_edges;
  for (std::size_t index = 0; index < quadratures.size(); ++index) {
    if (quadratures[index].integration !=
        SourceIntegration::isotropic_surface_shape_factor)
      continue;
    shape_factor_indices.push_back(index);
    if (quadratures[index].points.empty())
      throw std::runtime_error(
          "shape-factor source has no spatial quadrature points");
    const double dihedral =
        quadratures[index].shape_factor.feature_dihedral_degrees;
    if (!feature_edges.count(dihedral))
      feature_edges.emplace(dihedral, &impl_->feature_edges(dihedral));
  }

  const auto trace_shape_factor_quadrature =
      [&](std::size_t index) {
    thread_local DenseRowAccumulator row;
    row.reset(operators.transition.rows, operators.detection.cols,
              operators.losses.cols);
    double estimated_l1_error = 0.0;
    const auto& point_feature_edges = *feature_edges.at(
        quadratures[index].shape_factor.feature_dihedral_degrees);
    for (const auto& source_point : quadratures[index].points) {
      const auto candidates =
          impl_->domain_candidates.find(source_point.domain);
      if (candidates == impl_->domain_candidates.end())
        throw std::runtime_error(
            "shape-factor source domain has no boundary candidates");
      estimated_l1_error += trace_shape_factor_point(
          scene, geometry, primitive_to_state, channel_to_column,
          custom_runtimes,
          quadratures[index].shape_factor.backend ==
                  ShapeFactorBackend::generic_bvh
              ? nullptr
              : &impl_->structured_geometry,
          point_feature_edges, candidates->second,
          source_point,
          quadratures[index].shape_factor,
          operators.transition.rows, operators.detection.cols,
          operators.losses.cols, row);
    }
    result.source_integration_l1_error_estimate[index] = estimated_l1_error;
    row.store_to(result, index, operators.transition.rows,
                 operators.detection.cols, operators.losses.cols);
  };

  std::size_t worker_count = 1;
#ifdef _OPENMP
  worker_count = static_cast<std::size_t>(omp_get_max_threads());
#endif
  if (!shape_factor_indices.empty() &&
      shape_factor_indices.size() < worker_count) {
    // Small batches are faster source-by-source because each source can use
    // the complete worker pool for its much larger boundary-element loop.
    for (const auto index : shape_factor_indices)
      trace_shape_factor_quadrature(index);
  } else if (!shape_factor_indices.empty()) {
    // Likelihood searches submit many independent candidate coordinates.
    // Parallelize those candidates at the outer level; nested OpenMP regions
    // in trace_shape_factor_point are serialized by the production runtime.
    // This keeps the same quadrature and hard depth cap while bounding the
    // wall-clock tail of a batched response request.
    std::exception_ptr shape_factor_error;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t offset = 0;
         offset < static_cast<std::int64_t>(shape_factor_indices.size());
         ++offset) {
      try {
        trace_shape_factor_quadrature(shape_factor_indices[offset]);
      } catch (...) {
#pragma omp critical(oos_shape_factor_batch_error)
        {
          if (!shape_factor_error)
            shape_factor_error = std::current_exception();
        }
      }
    }
    if (shape_factor_error) std::rethrow_exception(shape_factor_error);
  }
  return result;
}

}  // namespace oos
