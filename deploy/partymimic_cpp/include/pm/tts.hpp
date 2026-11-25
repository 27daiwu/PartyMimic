// Non-blocking voice announcements through the G1 audio service (same AudioClient / TtsMaker / SetVolume usage as wbc_nx/remote_start).
// Runs only in the service process; announcements can never delay a control path (own worker thread, bounded queue).
#pragma once
#include <string>
#include <vector>

namespace pm {

// The ONLY phrases the program speaks (each <= 6 Chinese characters). The first 7 are the original set; the last 3 report the real exit result.
inline const std::vector<std::string>& tts_phrases() {
  static const std::vector<std::string> p = {"服务已启动", "正在加载", "准备就绪", "开始执行", "检查失败", "执行完成", "已退出",
                                             "异常已交还",   // abnormal exit, WalkRun recovery CONFIRMED (fsm in {500,801,802}, stable)
                                             "恢复失败",     // WalkRun recovery failed / robot already down -> final PASSIVE
                                             "紧急被动"};    // emergency PASSIVE channel (ESTOP file / remote L2+B)
  return p;
}
inline bool tts_known(const std::string& s) { for (auto& p : tts_phrases()) if (p == s) return true; return false; }

class Tts {
 public:
  Tts();
  ~Tts();
  // Needs an initialised unitree ChannelFactory. enabled=false -> text is only logged.
  void start(bool enabled, int volume);
  void say(const std::string& text);  // returns immediately
  void stop(double drain_s = 4.0);    // waits (bounded) for pending announcements, then joins the worker
 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace pm
