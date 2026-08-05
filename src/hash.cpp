#include "oos/hash.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace oos {
namespace {
constexpr std::array<std::uint32_t, 64> constants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
    0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
    0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
    0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
std::uint32_t rotate(std::uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}
class Sha256State {
 public:
  void update(const void* data_value, std::size_t size) {
    const auto* data = static_cast<const std::uint8_t*>(data_value);
    total_bytes_ += size;
    if (buffer_size_ != 0) {
      const auto copied = std::min(size, buffer_.size() - buffer_size_);
      std::memcpy(buffer_.data() + buffer_size_, data, copied);
      buffer_size_ += copied;
      data += copied;
      size -= copied;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
    }
    while (size >= buffer_.size()) {
      transform(data);
      data += buffer_.size();
      size -= buffer_.size();
    }
    if (size != 0) {
      std::memcpy(buffer_.data(), data, size);
      buffer_size_ = size;
    }
  }

  std::string finish() {
    const std::uint64_t bits = total_bytes_ * 8;
    std::array<std::uint8_t, 128> padding{};
    padding[0] = 0x80;
    const std::size_t padding_size =
        buffer_size_ < 56 ? 56 - buffer_size_ : 120 - buffer_size_;
    update(padding.data(), padding_size);
    std::array<std::uint8_t, 8> length{};
    for (int index = 0; index < 8; ++index)
      length[index] = static_cast<std::uint8_t>(bits >> (56 - 8 * index));
    update(length.data(), length.size());
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const auto value : hash_) result << std::setw(8) << value;
    return result.str();
  }

 private:
  void transform(const std::uint8_t* bytes) {
    std::array<std::uint32_t, 64> words{};
    for (int i = 0; i < 16; ++i)
      words[i] = (static_cast<std::uint32_t>(bytes[4 * i]) << 24) |
                 (static_cast<std::uint32_t>(bytes[4 * i + 1]) << 16) |
                 (static_cast<std::uint32_t>(bytes[4 * i + 2]) << 8) |
                 bytes[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const auto s0 = rotate(words[i - 15], 7) ^
                      rotate(words[i - 15], 18) ^
                      (words[i - 15] >> 3);
      const auto s1 = rotate(words[i - 2], 17) ^
                      rotate(words[i - 2], 19) ^
                      (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = hash_;
    for (int i = 0; i < 64; ++i) {
      const auto s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const auto choice = (e & f) ^ ((~e) & g);
      const auto temp1 = h + s1 + choice + constants[i] + words[i];
      const auto s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    hash_[0] += a;
    hash_[1] += b;
    hash_[2] += c;
    hash_[3] += d;
    hash_[4] += e;
    hash_[5] += f;
    hash_[6] += g;
    hash_[7] += h;
  }

  std::array<std::uint32_t, 8> hash_{
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t total_bytes_{};
};
}  // namespace

std::string sha256_string(std::string_view value) {
  Sha256State state;
  state.update(value.data(), value.size());
  return state.finish();
}
std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot hash " + path.string());
  Sha256State state;
  std::array<char, 1024 * 1024> buffer{};
  while (stream) {
    stream.read(buffer.data(), buffer.size());
    const auto count = stream.gcount();
    if (count > 0)
      state.update(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!stream.eof())
    throw std::runtime_error("failed while hashing " + path.string());
  return state.finish();
}
}  // namespace oos
