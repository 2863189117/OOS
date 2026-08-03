#include "oos/surface_behavior.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

#include "oos/physics.hpp"

namespace oos {
namespace {

Stokes scaled(const Stokes& value, double factor) {
  return {value.i * factor, value.q * factor, value.u * factor,
          value.v * factor};
}

void append(std::vector<LocalSurfaceBranch>& output, LocalBranchKind kind,
            const Stokes& value) {
  if (value.i > 0.0) output.push_back({kind, value});
}

}  // namespace

std::vector<LocalSurfaceBranch> evaluate_builtin_surface(
    const SurfaceModel& surface, const LocalSurfaceIncident& incident) {
  std::vector<LocalSurfaceBranch> output;
  if (surface.kind == SurfaceKind::specular_reflector) {
    append(output, LocalBranchKind::specular_reflection,
           scaled(incident.stokes, surface.reflectivity));
    append(output, LocalBranchKind::absorption,
           scaled(incident.stokes, 1.0 - surface.reflectivity));
    return output;
  }
  if (surface.kind == SurfaceKind::lambertian) {
    append(output, LocalBranchKind::lambertian_reflection,
           scaled(incident.stokes, surface.reflectivity));
    append(output, LocalBranchKind::absorption,
           scaled(incident.stokes, 1.0 - surface.reflectivity));
    return output;
  }
  if (surface.kind == SurfaceKind::sensitive) {
    append(output, LocalBranchKind::detection,
           scaled(incident.stokes, surface.detection_probability));
    const auto remainder =
        scaled(incident.stokes, 1.0 - surface.detection_probability);
    if (surface.remainder == RemainderAction::absorb)
      append(output, LocalBranchKind::absorption, remainder);
    else if (surface.remainder == RemainderAction::reflect_specular)
      append(output, LocalBranchKind::specular_reflection, remainder);
    else if (surface.remainder == RemainderAction::reflect_lambertian)
      append(output, LocalBranchKind::lambertian_reflection, remainder);
    else
      append(output, LocalBranchKind::straight_transmission, remainder);
    return output;
  }
  if (surface.kind != SurfaceKind::dielectric_fresnel)
    throw std::runtime_error("surface is not an intrinsic built-in behavior");

  const auto fresnel =
      fresnel_power(incident.incident_refractive_index,
                    incident.transmitted_refractive_index,
                    incident.cosine_incident);
  append(output, LocalBranchKind::specular_reflection,
         apply_linear_diattenuator(incident.stokes, fresnel.reflectance[0],
                                   fresnel.reflectance[1]));
  if (!fresnel.total_internal_reflection)
    append(output, LocalBranchKind::specular_transmission,
           apply_linear_diattenuator(incident.stokes,
                                     fresnel.transmittance[0],
                                     fresnel.transmittance[1]));
  return output;
}

std::vector<LocalSurfaceBranch> evaluate_custom_local_surface(
    const oos_local_interaction_v3& interaction,
    const LocalSurfaceIncident& incident, double tolerance) {
  const std::array<double, 5> probabilities{
      interaction.specular_reflection,
      interaction.specular_transmission,
      interaction.lambertian_reflection,
      interaction.detection,
      interaction.absorption};
  double total = 0.0;
  for (const auto probability : probabilities) {
    if (!std::isfinite(probability) || probability < 0.0)
      throw std::runtime_error(
          "local custom surface returned an invalid probability");
    total += probability;
  }
  if (std::abs(total - 1.0) > tolerance)
    throw std::runtime_error(
        "local custom surface probabilities do not close");
  std::vector<LocalSurfaceBranch> output;
  append(output, LocalBranchKind::specular_reflection,
         scaled(incident.stokes, interaction.specular_reflection));
  append(output, LocalBranchKind::specular_transmission,
         scaled(incident.stokes, interaction.specular_transmission));
  append(output, LocalBranchKind::lambertian_reflection,
         scaled(incident.stokes, interaction.lambertian_reflection));
  append(output, LocalBranchKind::detection,
         scaled(incident.stokes, interaction.detection));
  append(output, LocalBranchKind::absorption,
         scaled(incident.stokes, interaction.absorption));
  return output;
}

}  // namespace oos
