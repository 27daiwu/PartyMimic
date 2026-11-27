#include "pm/handback.hpp"

namespace pm {

HandbackReport handback(Link& link, const ExitRequest& req, const ExitConfig& cfg, ExitExecutor::FrameFn frames, ExitExecutor::PhaseFn phase, ExitExecutor::RemodeFn remode,
                        ExitExecutor::EmergencyFn emergency, ExitExecutor::LogFn log) {
  ExitExecutor ex(link, cfg, std::move(frames), std::move(phase), std::move(remode), std::move(emergency));
  if (log) ex.set_log(std::move(log));
  ex.start();
  ex.submit(req);
  while (!ex.wait(0.5)) {}
  HandbackReport r;
  r.req = ex.request();
  r.out = ex.outcome();
  ex.stop();
  return r;
}

}  // namespace pm
