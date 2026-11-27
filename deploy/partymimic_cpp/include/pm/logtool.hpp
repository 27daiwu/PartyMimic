// P5.1 offline log tools (src/logtool.cpp). No DDS, no robot.
#pragma once
#include <string>
#include <vector>

namespace pm {

struct FinalizeResult {
  std::string report, text, level = "OK";
  int n_fail = 0, n_warn = 0;
};
// Adds timing / safety_clips / logging / timeline / anomaly sections to run_report.json and writes anomaly_summary.{json,txt}.
// base_report: the runner's report text ("" = read <dir>/run_report.json; absent = salvage report "CRASH_NO_REPORT" rebuilt from the log files).
FinalizeResult finalize_run(const std::string& dir, const std::string& base_report, const std::string& note = "");
int logtool_check(const std::string& dir, bool quiet_pass);
int logtool_replay(const std::string& dir, const std::string& onnx, double tol);
std::vector<std::string> read_event_lines(const std::string& dir);

}  // namespace pm
