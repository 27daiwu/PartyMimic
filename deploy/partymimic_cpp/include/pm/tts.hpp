// Non-blocking voice announcements through the G1 audio service (same AudioClient / TtsMaker / SetVolume usage as wbc_nx/remote_start).
// Runs only in the service process; announcements can never delay a control path (own worker thread, bounded queue).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pm {

// The ONLY phrases the program speaks (each <= 6 Chinese characters). The first 7 are the original set; the next 3 report the real exit result; the last 6 name the selected test mode (P5.1).
inline const std::vector<std::string>& tts_phrases() {
  static const std::vector<std::string> p = {"服务已启动", "正在加载", "准备就绪", "开始执行", "检查失败", "执行完成", "已退出",
                                             "异常已交还",   // abnormal exit, WalkRun recovery CONFIRMED (fsm in {500,801,802}, stable)
                                             "恢复失败",     // WalkRun recovery failed / robot already down -> final PASSIVE
                                             "紧急被动",     // emergency PASSIVE channel (ESTOP file / remote L2+B)
                                             // P5.1: the selected test mode, said once after "服务已启动" (config/modes.conf column 'voice')
                                             "仅检查", "保持测试", "退出测试", "舞蹈二秒", "舞蹈四秒", "完整舞蹈",
                                             "可再次开始"};  // P5.3: said when a recoverable run has settled (WalkRun / attitude / lock / guardian confirmed) and the service is IDLE again - no ack needed. (V1.6.1: the 未授权 phrase is gone with the authorisation gate)
  return p;
}
inline bool tts_known(const std::string& s) { for (auto& p : tts_phrases()) if (p == s) return true; return false; }

// V1.6: what is known about one announcement. The G1 audio service has NO playback-finished callback (TtsMaker returns when the request is accepted; the SDK example sleeps 5 s afterwards):
//   finished = TtsMaker returned (rpc_done) + estimated speaking time. `est_done` is that conservative instant (mono()); the caller additionally applies its own hard upper bound (announce_max_s).
struct SayStatus {
  bool known = false;      // false: never queued (TTS not started) -> nothing to wait for
  bool dropped = false;    // pushed out of the bounded queue by newer announcements
  bool rpc_done = false;   // the TtsMaker RPC returned (or was skipped: audio unavailable)
  bool audio = false;      // a real audio client handled it
  int rc = 0;              // TtsMaker return code (0 = accepted)
  double t_say = 0, t_rpc_start = 0, t_rpc_end = 0, est_play_s = 0, est_done = 0;
};
inline int tts_char_count(const std::string& s) { int n = 0; for (unsigned char c : s) n += (c & 0xC0) != 0x80; return n; }

class Tts {
 public:
  Tts();
  ~Tts();
  // Needs an initialised unitree ChannelFactory. enabled=false -> text is only logged.
  void start(bool enabled, int volume, double est_s_per_char = 0.30);
  uint64_t say(const std::string& text);  // returns immediately; the id feeds status() (0 = not queued)
  SayStatus status(uint64_t id) const;
  void stop(double drain_s = 4.0);    // waits (bounded) for pending announcements, then joins the worker
 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace pm
