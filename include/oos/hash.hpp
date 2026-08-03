#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace oos {

std::string sha256_string(std::string_view value);
std::string sha256_file(const std::filesystem::path& path);

}  // namespace oos
