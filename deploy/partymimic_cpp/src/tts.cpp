#include "pm/tts.hpp"

#include <unitree/robot/g1/audio/g1_audio_client.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <algorithm>
#include <mutex>
#include <thread>

#include "pm/util.hpp"

namespace pm {

struct Tts::Impl {
  std::unique_ptr<unitree::robot::g1::AudioClient> audio;
  struct Item { std::string text; uint64_t id; };
  std::deque<Item> q;
  std::map<uint64_t, SayStatus> recs;  // last 32 announcements
  uint64_t next_id = 1;
  double busy_until = 0, est_s_per_char = 0.30;  // busy_until: estimated end of everything accepted so far (a new announcement is assumed to play after it)
  std::mutex mtx;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> running{false}, enabled{false};
  int volume = 100;
  size_t spoken = 0, queued = 0;
};

Tts::Tts() : impl_(new Impl) {}
Tts::~Tts() { stop(0.5); }  // impl_ is intentionally leaked: a detached worker may still be inside a blocking RPC

void Tts::start(bool enabled, int volume, double est_s_per_char) {
  Impl& I = *impl_;
  I.est_s_per_char = est_s_per_char;
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
      uint64_t id = 0;
      {
        std::unique_lock<std::mutex> lk(I.mtx);
        I.cv.wait(lk, [&] { return !I.q.empty() || !I.running; });
        if (I.q.empty()) return;  // not running and drained
        msg = I.q.front().text; id = I.q.front().id;
        I.q.pop_front();
        I.recs[id].t_rpc_start = mono();
      }
      int rc = 0;
      bool played = false;
      if (I.audio) {
        try {
          rc = I.audio->TtsMaker(msg.c_str(), 0);
          played = rc == 0;
          if (rc != 0) logf("[tts] TtsMaker rc=%d for \"%s\"", rc, msg.c_str());
        } catch (const std::exception& e) { rc = -1; logf("[tts] TtsMaker failed: %s", e.what()); }
      }
      std::lock_guard<std::mutex> lk(I.mtx);
      SayStatus& r = I.recs[id];
      r.t_rpc_end = mono(); r.rc = rc; r.audio = I.audio != nullptr; r.rpc_done = true;
      r.est_play_s = played ? tts_char_count(msg) * I.est_s_per_char + 0.3 : 0.0;
      r.est_done = played ? std::max(r.t_rpc_end, I.busy_until) + r.est_play_s : r.t_rpc_end;  // not played (no audio / rejected): nothing to wait for
      if (played) I.busy_until = r.est_done;
      I.spoken++;
    }
  });
}

uint64_t Tts::say(const std::string& text) {
  Impl& I = *impl_;
  logf("[TTS] %s%s", text.c_str(), tts_known(text) ? "" : "  (NOT in the phrase list!)");
  std::lock_guard<std::mutex> lk(I.mtx);
  if (!I.running) return 0;
  if (I.q.size() >= 4) { I.recs[I.q.front().id].dropped = true; I.q.pop_front(); I.queued--; }  // stale announcements are worthless: keep the newest (a dropped one was never spoken: not counted as pending)
  const uint64_t id = I.next_id++;
  I.q.push_back({text, id});
  SayStatus& r = I.recs[id];
  r.known = true; r.t_say = mono();
  while (I.recs.size() > 32) I.recs.erase(I.recs.begin());
  I.queued++;
  I.cv.notify_one();
  return id;
}

SayStatus Tts::status(uint64_t id) const {
  Impl& I = *impl_;
  std::lock_guard<std::mutex> lk(I.mtx);
  auto it = I.recs.find(id);
  return it == I.recs.end() ? SayStatus() : it->second;
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
