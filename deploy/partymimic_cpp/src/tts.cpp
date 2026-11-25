#include "pm/tts.hpp"

#include <unitree/robot/g1/audio/g1_audio_client.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "pm/util.hpp"

namespace pm {

struct Tts::Impl {
  std::unique_ptr<unitree::robot::g1::AudioClient> audio;
  std::deque<std::string> q;
  std::mutex mtx;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> running{false}, enabled{false};
  int volume = 100;
  size_t spoken = 0, queued = 0;
};

Tts::Tts() : impl_(new Impl) {}
Tts::~Tts() { stop(0.5); }  // impl_ is intentionally leaked: a detached worker may still be inside a blocking RPC

void Tts::start(bool enabled, int volume) {
  Impl& I = *impl_;
  I.enabled = enabled;
  I.volume = volume;
  I.running = true;
  I.worker = std::thread([&I]() {
    if (I.enabled) {
      try {
        I.audio.reset(new unitree::robot::g1::AudioClient());
        I.audio->SetTimeout(5.0f);
        I.audio->Init();
        int rc = I.audio->SetVolume((uint8_t)I.volume);
        logf("[tts] audio service ready (SetVolume(%d) rc=%d)", I.volume, rc);
      } catch (const std::exception& e) {
        logf("[tts] audio client init failed: %s (announcements are logged only)", e.what());
        I.audio.reset();
      }
    }
    for (;;) {
      std::string msg;
      {
        std::unique_lock<std::mutex> lk(I.mtx);
        I.cv.wait(lk, [&] { return !I.q.empty() || !I.running; });
        if (I.q.empty()) return;  // not running and drained
        msg = I.q.front();
        I.q.pop_front();
      }
      if (I.audio) {
        try {
          int rc = I.audio->TtsMaker(msg.c_str(), 0);
          if (rc != 0) logf("[tts] TtsMaker rc=%d for \"%s\"", rc, msg.c_str());
        } catch (const std::exception& e) { logf("[tts] TtsMaker failed: %s", e.what()); }
      }
      std::lock_guard<std::mutex> lk(I.mtx);
      I.spoken++;
    }
  });
}

void Tts::say(const std::string& text) {
  Impl& I = *impl_;
  logf("[TTS] %s%s", text.c_str(), tts_known(text) ? "" : "  (NOT in the phrase list!)");
  std::lock_guard<std::mutex> lk(I.mtx);
  if (!I.running) return;
  if (I.q.size() >= 4) I.q.pop_front();  // stale announcements are worthless: keep the newest
  I.q.push_back(text);
  I.queued++;
  I.cv.notify_one();
}

void Tts::stop(double drain_s) {
  Impl& I = *impl_;
  if (!I.worker.joinable()) return;
  double t_end = mono() + drain_s;
  while (mono() < t_end) {
    {
      std::lock_guard<std::mutex> lk(I.mtx);
      if (I.spoken >= I.queued) break;
    }
    sleep_s(0.05);
  }
  {
    std::lock_guard<std::mutex> lk(I.mtx);
    I.running = false;
    I.cv.notify_all();
  }
  // the worker may be inside a blocking RPC (<= timeout): do not hang the exit path for it
  I.worker.detach();
}

}  // namespace pm
