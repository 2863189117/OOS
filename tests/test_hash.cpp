#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "oos/hash.hpp"

TEST_CASE("SHA-256 matches the standard abc vector") {
  REQUIRE(oos::sha256_string("abc") ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad");
  const auto path = std::filesystem::temp_directory_path() / "oos-hash-test";
  {
    std::ofstream output(path, std::ios::binary);
    output << "abc";
  }
  REQUIRE(oos::sha256_file(path) == oos::sha256_string("abc"));
  std::filesystem::remove(path);
}
