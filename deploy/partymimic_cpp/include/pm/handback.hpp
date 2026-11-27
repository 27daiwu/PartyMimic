// Synchronous hand-back used by the guardian and `partymimic release`: a thin wrapper around the SAME ExitExecutor / decision table the runner session uses
// (pm/exit_policy.hpp). There is no separate hand-back logic any more: confirmed = GetFsmId code 0 AND fsm in the requested mode's expected set
// (WALKRUN family {500,801,802}, PASSIVE = 1), still true after dwell + a stable observation window. RPC success with a mismatching fsm is NEVER a success.
#pragma once
#include "pm/exit_policy.hpp"

namespace pm {

struct HandbackReport {
  ExitRequest req;
  ExitOutcome out;
};

// Blocks until the outcome is final (confirmed / settled on the PASSIVE fallback / unconfirmed after the hold time).
HandbackReport handback(Link& link, const ExitRequest& req, const ExitConfig& cfg, ExitExecutor::FrameFn frames, ExitExecutor::PhaseFn phase = nullptr,
                        ExitExecutor::RemodeFn remode = nullptr, ExitExecutor::EmergencyFn emergency = nullptr, ExitExecutor::LogFn log = nullptr);

}  // namespace pm
