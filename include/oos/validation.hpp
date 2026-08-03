#pragma once

#include <string>
#include <vector>

#include "oos/scene.hpp"

namespace oos {

struct ValidationIssue {
  enum class Severity { warning, error };
  Severity severity{};
  std::string code;
  std::string message;
};

struct ValidationReport {
  std::vector<ValidationIssue> issues;
  bool ok() const;
  void throw_if_invalid() const;
};

class SceneValidator {
 public:
  static ValidationReport validate(const Scene& scene);
};

}  // namespace oos
