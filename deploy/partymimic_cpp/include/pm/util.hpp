// Small utilities: monotonic clock, thread-safe logging, key/value config files, atomic file write, SHA-256, CRC32 (Unitree variant).
#pragma once
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace pm {

inline double mono() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double wall() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
void sleep_s(double s);

// ---- logging (stdout + optional file; every line flushed so a crash keeps everything)
void log_set_file(const std::string& path);
void log_set_tag(const std::string& tag);
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string timestamp_compact();  // 20261124_170000

// ---- files
bool read_file(const std::string& path, std::string& out);
bool write_atomic(const std::string& path, const std::string& content);
bool file_exists(const std::string& path);
std::string json_escape(const std::string& s);

// ---- key/value config: "key v1 v2 ..." per line, '#' comments. Used for config/site.conf and assets/policy_params.txt
struct KV {
  std::map<std::string, std::vector<std::string>> m;
  bool load(const std::string& path);
  bool has(const std::string& k) const { return m.count(k) != 0; }
  std::string str(const std::string& k, const std::string& def = "") const;
  double num(const std::string& k, double def) const;
  std::vector<double> nums(const std::string& k) const;
  std::vector<int> ints(const std::string& k) const;
};

// ---- hashing
std::string sha256_hex(const void* data, size_t n);
bool sha256_file(const std::string& path, std::string& hex);
// Unitree HG LowCmd CRC: poly 0x04c11db7, init 0xffffffff, no reflection, no final xor, over 32-bit words
uint32_t crc32_words(const uint32_t* ptr, uint32_t len);

}  // namespace pm
