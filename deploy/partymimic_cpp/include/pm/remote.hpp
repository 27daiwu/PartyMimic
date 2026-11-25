// Remote-controller decoding, chords and START double-click (pure logic, no I/O).
// KEY LAYOUT = the one already used and verified on the robot by wbc_nx/remote_start: `rt/lowstate.wireless_remote` (40 bytes) is read through
// unitree::common::REMOTE_DATA_RX / xKeySwitchUnion (include/gamepad.hpp, copied from remote_start unchanged). Bit order of btn.value:
//   0 R1, 1 L1, 2 START, 3 SELECT, 4 R2, 5 L2, 6 F1, 7 F2 (called F3 on our remote: same bit), 8 A, 9 B, 10 X, 11 Y, 12 UP, 13 RIGHT, 14 DOWN, 15 LEFT
// Rules taken over from remote_start: edge triggered (no auto-repeat while held); a function needs EXACTLY its keys down, any extra key cancels it.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "gamepad.hpp"

namespace pm {

enum KeyBit : uint16_t {
  K_R1 = 1u << 0, K_L1 = 1u << 1, K_START = 1u << 2, K_SELECT = 1u << 3, K_R2 = 1u << 4, K_L2 = 1u << 5, K_F1 = 1u << 6, K_F3 = 1u << 7,
  K_A = 1u << 8, K_B = 1u << 9, K_X = 1u << 10, K_Y = 1u << 11, K_UP = 1u << 12, K_RIGHT = 1u << 13, K_DOWN = 1u << 14, K_LEFT = 1u << 15
};
constexpr uint16_t COMBO_LOAD = K_F1 | K_F3;  // F1+F3: load Policy + real-time checks
constexpr uint16_t COMBO_GO = K_L1 | K_A;     // L1+A : execute
constexpr uint16_t COMBO_DAMP = K_L2 | K_B;   // L2+B : factory damping chord (keys.log: byte2=0x20 byte3=0x02) -> emergency PASSIVE during the run

inline uint16_t keys_from_wireless(const std::array<uint8_t, 40>& w) {
  unitree::common::REMOTE_DATA_RX rx;
  std::memcpy(rx.buff, w.data(), 40);
  return rx.RF_RX.btn.value;
}
inline bool start_pressed(uint16_t keys) { return (keys & K_START) != 0; }
inline std::string keys_text(uint16_t k) {
  static const char* n[16] = {"R1", "L1", "START", "SELECT", "R2", "L2", "F1", "F3", "A", "B", "X", "Y", "UP", "RIGHT", "DOWN", "LEFT"};
  std::string s;
  for (int i = 0; i < 16; i++) if (k & (1u << i)) { if (!s.empty()) s += "+"; s += n[i]; }
  return s.empty() ? "(none)" : s;
}

// feed(t, pressed) -> true exactly when the 2nd rising edge arrives within window_s of the 1st (the SDK's own double-click rule)
class StartDoubleClick {
 public:
  explicit StartDoubleClick(double window_s = 0.3, double min_gap_s = 0.03) : window_(window_s), min_gap_(min_gap_s) {}
  bool feed(double t, bool pressed) {
    bool fired = false;
    if (pressed && !prev_) {
      edges++;
      if (have_edge_ && t - last_edge_ <= window_ && have_release_ && t - last_release_ >= 0.0 && t - last_edge_ >= min_gap_) { fired = true; have_edge_ = false; }
      else { last_edge_ = t; have_edge_ = true; }
    }
    if (!pressed && prev_) { last_release_ = t; have_release_ = true; }
    prev_ = pressed;
    return fired;
  }
  int edges = 0;
 private:
  double window_, min_gap_, last_edge_ = 0, last_release_ = 0;
  bool prev_ = false, have_edge_ = false, have_release_ = false;
};

// fires ONCE when the pressed set equals exactly `combo` for >= hold_s; re-arms only after ALL keys were released for >= release_s
class ChordDetector {
 public:
  explicit ChordDetector(uint16_t combo, double hold_s = 0.15, double release_s = 0.2) : combo_(combo), hold_(hold_s), release_(release_s) {}
  bool feed(double t, uint16_t pressed) {
    if (pressed == 0) {
      if (!have_free_) { free_since_ = t; have_free_ = true; }
      if (t - free_since_ >= release_) { armed_ = true; fired_ = false; }
      have_since_ = false;
      return false;
    }
    have_free_ = false;
    if (pressed == combo_ && armed_ && !fired_) {
      if (!have_since_) { since_ = t; have_since_ = true; }
      if (t - since_ >= hold_) { fired_ = true; armed_ = false; return true; }
    } else have_since_ = false;
    return false;
  }
 private:
  uint16_t combo_;
  double hold_, release_, since_ = 0, free_since_ = 0;
  bool have_since_ = false, have_free_ = false, fired_ = false, armed_ = true;
};

// Emergency PASSIVE chord. Fires ONCE when L2 and B are both down continuously for >= hold_s (and >= min_samples distinct states), but ONLY while `active`
// (control taken) and only for a press that STARTED while active (keys already held when control began are ignored until released: no accidental trip from the
// L1+A / F1+F3 hand-over). Extra keys do not cancel it (an emergency must not be defeatable by a stray finger). Deactivating resets everything.
class DampCombo {
 public:
  explicit DampCombo(double hold_s = 0.10, int min_samples = 2) : hold_(hold_s), min_samples_(min_samples) {}
  bool feed(double t, uint16_t keys, bool active) {
    const bool pressed = (keys & COMBO_DAMP) == COMBO_DAMP;
    if (!active) { armed_ = false; fired_ = false; since_ = 0; n_ = 0; return false; }
    if (!pressed) { armed_ = true; since_ = 0; n_ = 0; return false; }
    if (!armed_ || fired_) return false;
    if (since_ == 0) since_ = t > 0 ? t : 1e-9;
    n_++;
    if (n_ >= min_samples_ && t - since_ >= hold_) { fired_ = true; return true; }
    return false;
  }
 private:
  double hold_;
  int min_samples_;
  double since_ = 0;
  int n_ = 0;
  bool armed_ = false, fired_ = false;
};

}  // namespace pm
