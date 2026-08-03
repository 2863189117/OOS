#pragma once

#include <vector>

#include "oos/plugin.h"
#include "oos/types.hpp"

namespace oos {

// A surface behavior may decide how much radiance leaves through each local
// branch, but it never names a destination primitive, transport state, or
// sensitive channel.  Geometry assembly is deliberately a separate step.
enum class LocalBranchKind {
  specular_reflection,
  specular_transmission,
  straight_transmission,
  lambertian_reflection,
  detection,
  absorption,
};

struct LocalSurfaceBranch {
  LocalBranchKind kind{};
  Stokes stokes{};
};

struct LocalSurfaceIncident {
  Stokes stokes{};
  double incident_refractive_index{};
  double transmitted_refractive_index{};
  double cosine_incident{};
};

std::vector<LocalSurfaceBranch> evaluate_builtin_surface(
    const SurfaceModel& surface, const LocalSurfaceIncident& incident);

std::vector<LocalSurfaceBranch> evaluate_custom_local_surface(
    const oos_local_interaction_v3& interaction,
    const LocalSurfaceIncident& incident, double tolerance);

}  // namespace oos
