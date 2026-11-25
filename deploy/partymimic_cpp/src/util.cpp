#include <algorithm>
#include "pm/util.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <thread>

namespace pm {

void sleep_s(double s) {
  if (s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

static std::mutex g_log_mtx;
static FILE* g_log_file = nullptr;
static std::string g_tag;

void log_set_file(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_log_mtx);
  if (g_log_file) fclose(g_log_file);
  g_log_file = path.empty() ? nullptr : fopen(path.c_str(), "a");
}
void log_set_tag(const std::string& tag) {
  std::lock_guard<std::mutex> lk(g_log_mtx);
  g_tag = tag;
}
void logf(const char* fmt, ...) {
  char buf[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  time_t t = time(nullptr);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char ts[16];
  strftime(ts, sizeof ts, "%H:%M:%S", &tmv);
  std::lock_guard<std::mutex> lk(g_log_mtx);
  std::fprintf(stdout, "%s %s%s%s\n", ts, g_tag.c_str(), g_tag.empty() ? "" : " ", buf);
  std::fflush(stdout);
  if (g_log_file) {
    std::fprintf(g_log_file, "%s %s%s%s\n", ts, g_tag.c_str(), g_tag.empty() ? "" : " ", buf);
    std::fflush(g_log_file);
  }
}
std::string timestamp_compact() {
  time_t t = time(nullptr);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char ts[32];
  strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", &tmv);
  return ts;
}

bool read_file(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}
bool write_atomic(const std::string& path, const std::string& content) {
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f << content;
    if (!f) return false;
  }
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}
bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}
std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if ((unsigned char)c < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

bool KV::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    size_t h = line.find('#');
    if (h != std::string::npos) line.resize(h);
    std::istringstream ss(line);
    std::string k, v;
    if (!(ss >> k)) continue;
    std::vector<std::string> vals;
    while (ss >> v) vals.push_back(v);
    m[k] = vals;
  }
  return true;
}
std::string KV::str(const std::string& k, const std::string& def) const {
  auto it = m.find(k);
  return (it == m.end() || it->second.empty()) ? def : it->second[0];
}
double KV::num(const std::string& k, double def) const {
  auto it = m.find(k);
  if (it == m.end() || it->second.empty()) return def;
  try { return std::stod(it->second[0]); } catch (...) { return def; }
}
std::vector<double> KV::nums(const std::string& k) const {
  std::vector<double> o;
  auto it = m.find(k);
  if (it == m.end()) return o;
  for (auto& s : it->second) o.push_back(std::stod(s));
  return o;
}
std::vector<int> KV::ints(const std::string& k) const {
  std::vector<int> o;
  auto it = m.find(k);
  if (it == m.end()) return o;
  for (auto& s : it->second) o.push_back(std::stoi(s));
  return o;
}

// ---------------- SHA-256
namespace {
struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf[64];
  uint64_t len = 0;
  size_t fill = 0;
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g), t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    len += n;
    while (n) {
      size_t k = std::min(n, 64 - fill);
      memcpy(buf + fill, p, k);
      fill += k; p += k; n -= k;
      if (fill == 64) { block(buf); fill = 0; }
    }
  }
  std::string finish() {
    uint64_t bits = len * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t z = 0;
    while (fill != 56) update(&z, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(lb, 8);
    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
    return out;
  }
};
}  // namespace

std::string sha256_hex(const void* data, size_t n) {
  Sha256 s;
  s.update(data, n);
  return s.finish();
}
bool sha256_file(const std::string& path, std::string& hex) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  Sha256 s;
  std::vector<char> b(1 << 20);
  while (f) {
    f.read(b.data(), b.size());
    s.update(b.data(), (size_t)f.gcount());
  }
  hex = s.finish();
  return true;
}

uint32_t crc32_words(const uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF;
  const uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    uint32_t xbit = 1u << 31, data = ptr[i];
    for (int b = 0; b < 32; b++) {
      if (crc & 0x80000000) { crc <<= 1; crc ^= poly; } else crc <<= 1;
      if (data & xbit) crc ^= poly;
      xbit >>= 1;
    }
  }
  return crc;
}

}  // namespace pm
