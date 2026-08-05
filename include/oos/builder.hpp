#pragma once

#include "oos/scene.hpp"
#include "oos/solver.hpp"
#include "oos/source.hpp"

namespace oos {

class OperatorBuilder {
 public:
  static std::string cache_key(const Scene& scene);
  static OperatorSet build(const Scene& scene);
};

}  // namespace oos
