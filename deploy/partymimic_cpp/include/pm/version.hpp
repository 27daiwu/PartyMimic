// P5.1 program identity (recorded in manifest.json / run_report.json of every run next to the sha256 of the executable and of every frozen artifact)
#pragma once
namespace pm {
constexpr const char* PM_VERSION = "V1.6.1";
constexpr const char* PM_POLICY_NAME = "robust_d_it12000";  // frozen: no training / parameter search any more (sha256 in frozen.hpp)
constexpr const char* PM_LOG_SCHEMA = "pm-run-log-3";  // 3: + trace columns dance_frame / dance_t / seg (appended: older readers keep working), log_stats_ver 2
constexpr const char* PM_BUILD_STAMP = __DATE__ " " __TIME__;
}  // namespace pm
