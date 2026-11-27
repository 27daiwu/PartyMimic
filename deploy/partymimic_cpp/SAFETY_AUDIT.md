# SAFETY_AUDIT.md — 软件安全门审计与放宽

范围：`deploy/partymimic_cpp` 全部软件门（代码里每一个"会拒绝 / 会裁剪 / 会触发退出"的数值）。
审计原则（用户要求）：

* **保留**：通信中断、非法数值（NaN/Inf/四元数范数）、**硬件极限**（关节位置/速度/力矩的硬件规格，本次不放宽，只允许收紧）、紧急停止。
* **放宽**：只为"保护"而**裁剪 Policy 动作**的软件限制（目标位置内缩 margin、目标限速 0.6×、PD 力矩 0.85×），放到贴近硬件规格。
* 所有阈值都是 `GuardConfig` / `ExitConfig` / `SessionConfig` / `Site` 字段，可由 `config/site.conf` 覆盖；**范围检查强制**（越界 = runner 拒绝启动，见 `validate()`）。硬件类阈值的允许范围上界就是硬件规格本身，无法在 site.conf 里放宽到规格之外。
* **P4 更新**：默认改为 `run_s=3.0`（hold）、`authority=1.0`、`ramp_s=0.3`，caps `max_run_s=12`/`max_authority=1.0`（authority 的编译期上限同；**run_s 的编译期上限已在 P6 由 12 s 提高到 60 s，默认 `max_run_s` 仍为 12**，见 R1）；依据 Sim2Sim（`results/p4/P4_REPORT.md` §2）：authority 0.3 是不稳定区。**该包络从未上过真机**：任何 authority>0.3 或窗口>1.0 s 的真机运行需 `--accept-unvalidated`（start.sh 要求键入 `I-ACCEPT-UNVALIDATED`；runner 在 DDS 之前拒绝，`site.conf` 无法授予）；持续 authority∈(0.05,0.9) 拒绝（`--allow-partial-authority`）。
* `assets/`（policy.onnx、motion.bin、policy_params.txt）、`src/policy.cpp` 数值路径、runner 里 `q_cmd = q_hold + authority*(q_target-q_hold)` 控制律均未改动。硬件速度/力矩规格取自冻结的 `policy_params.txt`（`vel_lim`、`effort_lim`、`pos_lo/pos_hi`）。

分类缩写：**HW保护**（硬件保护）· **通信** · **数值**（非法数值）· **HW极限**（硬件极限）· **急停**（紧急停止）· **过保守**（过度保守的软件限制）· **入场**（入场准入，保护性闸门，非裁剪动作）。

## 1. GuardConfig（`include/pm/guard.hpp`，`src/guard.cpp`；site.conf 同名键）

| # | 字段 | 位置 | 原值 | 分类 | 结论 / 新值 / 理由 |
|---|---|---|---|---|---|
| G1 | `state_stale_s` | `check_state`：lowstate 年龄 > 值 → `COMM_TIMEOUT` | 0.060 s | 通信 | **保留**。通信中断判据；允许范围 [0.02, 0.2]。（看门狗线程另用 2× 此值判"lowstate 沉默"） |
| G2 | `state_frozen_ticks` | tick 计数连续不变 → `STATE_FROZEN` | 5 | 通信 | **保留**；[2, 20] |
| G3 | `limit_fault_tol_rad` | 测得 q 超出硬限位 + 值 → `JOINT_LIMIT` | 0.05 rad | HW极限 | **保留**（硬件极限本身不放宽）；允许范围只能收紧 [0, 0.05] |
| G4 | `speed_fault_frac` | 测得 \|dq\| > 值×硬件速度限 → `JOINT_SPEED` | 0.90 | HW保护 | **保留 0.90**：已是"接近硬件极限"的故障门（不裁剪 Policy，只在测得超速时退出）；范围 [0.5, 1.0]，不能 >1 |
| G5 | `unknown_speed_limit` | 无已知速度限的关节（腰 roll/pitch）用的速度限 | 20 rad/s | HW极限 | **保留**；[1, 20]，不能放宽 |
| G6 | `torque_meas_fault_frac` / `_ticks` | 测得 \|tau_est\| > 值×硬件力矩限且连续 N tick → `TORQUE_SUSTAINED` | 1.20 / 10（P4 由 1.00 / 8 调整） | HW保护 | **保留但略放宽**：Sim2Sim 中正常舞蹈在髋 roll 上有最长 10 tick（0.2 s）的力矩饱和，测得 tau_est 会在额定峰值附近抖动；1.00×8 tick 会把正常动作误判为超载。改为 >1.2×额定峰值连续 10 tick（0.2 s）；[0.5, 1.5] / [1, 50] |
| G7 | `max_tilt_rad` | 倾角 > 值 → `TILT`（**倒地**） | 1.0 rad | HW保护 / 急停类 | **保留 1.0 rad，语义改为"已倒地 → PASSIVE"**；新增更低一级的失稳门（见 §4，`ExitConfig.unstable_*`）。范围 [0.6, 1.5] |
| G8 | `max_gyro_rad_s` | \|gyro\| > 值 → `GYRO` | 15 rad/s（P4 由 10 调整） | HW保护 / 数值 | **放宽**：Sim2Sim 正常舞蹈 |gyro| 峰值已达 7.9 rad/s（含全套扰动 ≤8），10 rad/s 余量过小；15 rad/s 仍远低于 IMU 量程，只拦传感器失真/剧烈翻滚；[2, 25]。触发 → 立即 WALKRUN（若未倒地） |
| G9 | 状态有限性 + 四元数范数 0.9~1.1 | `check_state` | 固定 | 数值 | **保留**，不可配置（非法数值） |
| G10 | `pos_margin_rad` | `filter_command`：目标位置被夹到 [lo+m, hi−m] | **0.02** rad | 过保守 | **放宽 → 0.005 rad**。0.02 rad 的内缩会吞掉 Policy 在极限附近的动作（P2 记录过限位问题，真正的限位保护由 G3 兜底）；0.005 仅保留数值/编码器余量。范围 [0, 0.1]。硬限位本身不变 |
| G11 | `target_speed_frac` | 目标 slew 限速 = 值 × 硬件速度限 × dt | **0.60** | 过保守 | **放宽 → 1.0**（= 硬件速度规格）。0.6× 会在快动作处把 Policy 目标拖慢，造成与参考不一致；超速由 G4 的测得速度门兜底。范围 [0.1, 1.0] |
| G12 | `torque_cmd_frac` | PD 力矩 `kp·e−kd·dq` 限制在 值×硬件力矩限（通过缩小位置误差实现） | **0.85** | 过保守 | **放宽 → 1.0**（= 硬件力矩规格）。风险记录：指令贴近 100% 时测得力矩更容易越过 G6 → `TORQUE_SUSTAINED`；这是真实超载信号，保留。范围 [0.1, 1.0] |
| G13 | `diverge_rad` / `diverge_ticks`（原 runner 内 1.5×8 硬编码） | runner：\|q−q_cmd\| 连续 N tick 超值 → `NO_RESPONSE` | 1.5 rad × 8 | HW保护（指令未被跟随） | **保留，改为 GuardConfig 字段**可由 site.conf 覆盖；CLI `--diverge-rad` 仍可覆盖；[0.3,3.0] / [2,50] |
| G14 | `loop_overrun_factor`（原 `3×dt` 硬编码） | runner：控制步耗时 > 值×dt → `LOOP_OVERRUN` | 3.0 | 通信（控制环健康） | **保留，改为字段**；[1.5, 5] |
| G15 | kp 不超过 yaml、负增益拒绝、dq_target=0 / tau_ff=0 强制、非有限指令拒绝 | `filter_command` | 固定 | 数值 / HW保护 | **保留**（非法数值与"Policy 不得越权写前馈/速度目标"） |

## 2. SessionConfig / 入场准入 / 看门狗（`include/pm/session.hpp`，`src/session.cpp`）

| # | 项 | 原值 | 分类 | 结论 / 新值 / 理由 |
|---|---|---|---|---|
| S1 | `watchdog_s`（控制环心跳） | 0.10 s | 通信 | **保留**，site.conf 可配 [0.05, 0.5]；超时 → `WATCHDOG` → 立即 WALKRUN |
| S2 | lowstate 沉默（看门狗线程） | 2×`state_stale_s` = 120 ms | 通信 | **保留**（派生值） |
| S3 | `min_state_hz`（入场：≥30 Hz PASS，15–30 Hz WARN，<15 FAIL） | 30 | 通信 | **保留**，site.conf 可配 [15, 500] |
| S4 | tick 推进（入场） | >5 个不同 tick PASS | 通信 | **保留** |
| S5 | `quiescent_dq`（入场静止度：<0.5 PASS，<4× WARN，其余 FAIL） | 0.5 rad/s | 入场 | **保留**（站稳才接管），site.conf 可配 [0.1, 2.0]；真实机器人上 WARN 不阻断 |
| S6 | `max_entry_tilt_rad`（入场倾角） | 0.5 rad | 入场 | **保留**（跌倒保护；接管一个已倾斜 0.5 rad 的机器人没有意义），site.conf 可配 [0.1, 0.8]；与失稳门 0.40 / 倒地门 1.0 一致 |
| S7 | `entry_fsm_allowed`（0/1 恒拒绝；其余必须在列表内） | 500 505 801 802 | 入场 | **保留**；0/1 恒拒绝是硬编码安全规则 |
| S8 | q 在硬限位内、状态在 guard 包络内、mode_machine ∈ {0,5} | — | HW极限 / 数值 | **保留** |
| S9 | 有源竞争写者检测（`rt/user_lowcmd`、`rt/arm_sdk` 窗口内有样本）、独占 flock、急停文件 | listen 1.0 s（GO 前 0.3 s） | HW保护 / 急停 | **保留**（控制权互斥）。writer *数量*不作判据（原厂有空闲 writer） |
| S10 | `confirm_timeout_s`（SwitchToUserCtrl 确认） | 3.0 s | 通信 | **保留** |
| S11 | FSM 轮询（每 1 s，连续 2 次不等于 user fsm → `FSM_CHANGED`） | 1 s | 通信 / HW保护 | **保留** |
| S12 | 外来写者（rx−sent>15）→ `LOCK_LOST` | 15 | HW保护 | **保留** |
| S13 | 急停文件 `/tmp/partymimic_cpp_ESTOP`（3 ms 线程 + write() 内检查） | — | 急停 | **保留，且语义改为直接 PASSIVE**（不经 WalkRun） |
| S14 | **新增** 遥控器 L2+B（`l2b_hold_s`=0.10 s 去抖，仅运行期） | — | 急停 | 新增独立紧急 PASSIVE 通道 |
| S15 | 退出确认时序：`damp_before_release_s`=0.1、`attitude_wait_s`=2.0、`handback_dwell_s`、`unconfirmed_exit_hold_s`、`confirm_timeout_s` | — | 过保守（拖慢交还） | **重构**：先 damp 0.1 s、等姿态的流程删除；改为 `ExitConfig`（立即 RPC、HOLD 帧、dwell 0.3 s + 稳定窗口 0.8 s、未确认保持 30 s），见 `EXIT_POLICY.md` |

## 3. Runner / Service / Guardian 常量

| # | 项 | 位置 | 原值 | 分类 | 结论 / 新值 / 理由 |
|---|---|---|---|---|---|
| R1 | `REAL_MAX_RUN_S` | `app.hpp` / runner 硬拒绝 | 硬编码 1.0 s | 过保守（硬编码） | **改为 site.conf `max_run_s`（P4 默认 12）+ 编译期绝对上限 `ABS_MAX_RUN_S`（P4：12 s；P6：**60 s**，只是天花板——默认 `max_run_s` 仍 12，不会因此放宽任何运行；比 12 s 更长的舞蹈由 `./pm package` 在该包的 `site.conf` 写入显式 `max_run_s` 覆盖并记入 `ADMISSION.json`，仍需 `--accept-unvalidated`；`site.conf max_run_s>60`、模式表 `run_s>60`、`--sim --run-s>60` 一律拒绝）**。run_s 超过旧上限 1.0 时：runner 日志 WARNING、`ready.json.exceeds_legacy_caps=true`、admission 项 `run_caps` = WARN（含 "EXCEEDS THE LEGACY CAPS"）、run_report 同名字段、service 启动日志 WARNING | P4：超出验证包络需 `--accept-unvalidated`。
| R2 | `REAL_MAX_AUTHORITY` | 同上 | 硬编码 0.3 | 过保守（硬编码） | **改为 `max_authority`（P4 默认 1.0；真机需 `--accept-unvalidated`；0.05<a<0.9 拒绝）+ `ABS_MAX_AUTHORITY=1.0`**，标注同上 |
| R3 | `RAMP_S`（authority 0→max 的 smoothstep） | runner | 0.1 s | 过保守 / 控制律邻近 | **不动**（属控制律段，由你接手） |
| R4 | 控制步 50 Hz、`el > run_s+1.0` 兜底 | runner | — | 通信 | 保留 |
| R5 | 冻结文件 sha256、guardian 心跳 <1 s、iface≠lo（真机）、`--i-confirm-protection` | runner | — | HW保护 | 保留 |
| R6 | 推理耗时 >15 ms（WARN，不阻断） | runner | 15 ms | 通信 | 保留 |
| R7 | RPC 超时 | Link | 2.0 s（guardian 1.0 s） | 通信 | 保留 |
| S-V1 | service：`run_max_s = run_s+60`、`abort_escalate 2.5 s`、`load_timeout 90`、GO 键松开 0.25 s / 3 s | service | — | 流程 | 保留 |
| G-V1 | guardian：`--silence-s 0.30`、`--remote-grace-s 0.25`、新增 `--estop-grace-s 0.10`、`exit_owner_timeout_s=10` | guardian | — | 通信 / 急停 | 保留；新增"runner 正在退出则不并发"闸门 |
| E-V1 | `DAMP_KD=3`（PASSIVE 路径阻尼帧） | profile.hpp | 3.0 | HW保护 | 保留 |

## 4. TILT 拆成两级（`ExitConfig`，`src/exit_policy.cpp`）

| 级别 | 条件（默认，可配） | 退出 |
|---|---|---|
| 失稳 `UNSTABLE` | tilt > `unstable_tilt_rad`=0.40，**或** tilt > `unstable_soft_tilt_rad`=0.30 且（\|gyro_xy\| > `unstable_gyro_rad_s`=8.0 或 d(tilt)/dt > `unstable_tilt_rate_rad_s`=3.0），连续 `unstable_debounce_samples`=2 个不同 lowstate 样本 | **立即 WALKRUN** |
| 倒地 `TILT` | tilt > `fallen_tilt_rad` = `max_tilt_rad` = 1.0 rad | **PASSIVE** |

### V1.6 A1：失稳门改为"参考感知" + 绝对硬上限（`ExitConfig`，`src/exit_policy.cpp`）

真机 DIY 舞蹈 3 次在 ≈7.4 s 被固定 0.40 失稳门退出（机器人在跟随一个本身前倾 0.40 rad 的参考；整段参考最大 0.498 rad）。固定门与舞蹈冲突，因此：

| 门 | 条件（默认，可配，范围检查） | 依赖参考？ | 分类 |
|---|---|---|---|
| 绝对硬上限 `unstable_hard_tilt_rad` | tilt > **0.70** → 立即 WALKRUN | **否**（不被参考抬高；参考通道无效/缺失/NaN/过期也生效） | 保护（新增，独立） |
| 参考偏离 `ref_dev_rad` | 重力方向误差 > **0.35** 且 tilt > 0.25 且误差未在缩小；参考有效且 ≤0.15 s 新鲜 | 是 | 保护（取代参考运行期的固定 0.40） |
| 固定失稳门 `unstable_tilt_rad` | tilt > 0.40，**仅在没有有效参考时**（前导/paused/保持/缺失/NaN/过期） | 否 | 保护（保守回退） |
| 软门（原有） | tilt > 0.30 且（\|gyro_xy\| > 8 或 倾角率 > 3） | 否 | 保护（保留） |
| 倒地 `fallen_tilt_rad` / guard `max_tilt_rad` | tilt > 1.0 → PASSIVE；guard `max_gyro_rad_s` 15 | 否 | 保护（不变） |

`ref_aware_unstable 0` 恢复 V1.5.1 行为。风险：参考感知门依赖 runner 正确发布参考（`set_reference` 每 tick，`valid` 仅在舞蹈运行期）；发布停止 0.15 s 后自动回退到 0.40，且 0.70 硬上限始终独立。详见 EXIT_POLICY.md §3b 与 `results/diy/v16/A_deploy/TILT_JUDGE_REPORT.md`。

检测运行在 3 ms 看门狗线程（最新 lowstate 样本变化即评估）和 runner 控制环两处。数值是起点，等你用 sim 复现后微调（范围检查：`unstable_tilt_rad ∈ [0.1, fallen)`，`soft ≤ unstable`，等）。

## 5. 放宽后残余风险

1. `torque_cmd_frac=1.0` + `torque_meas_fault_frac=1.0`：真正饱和的动作会以 `TORQUE_SUSTAINED` 退出（→ WALKRUN）。这是期望的"测得超载才退出"。
2. `target_speed_frac=1.0`：目标 slew 不再把快动作拖慢；执行器真实速度仍受硬件限制，测得 \|dq\|>0.9×限 → `JOINT_SPEED`。
3. `pos_margin_rad=0.005`：目标可以非常靠近硬限位；测得 q 越界 0.05 rad 才是 `JOINT_LIMIT`。
4. 以上三项在 sim_robot 里不受力矩/速度物理限制，**未做真机验证**，需要你离线验证后再调默认。
5. P4：caps 已提高到 12 s / 1.0（P6：run_s 编译期天花板 60 s，默认 cap 不变），默认 `authority=1.0`；真机需每次键入 `I-ACCEPT-UNVALIDATED`（`--accept-unvalidated`）。若要回到旧 30% 混合做对照，必须同时加 `--allow-partial-authority`（不稳定区，仅实验）。

## P5.1：安全裁剪的记录（阈值与行为不变）
`SafetyGuard::filter_command` 可选返回 `FilterTrace`：每个关节在 POS（限位内缩）/ SLEW（限速）/ TORQUE（PD 力矩上限）/ POS2（力矩后再夹限位）/ KP（kp 上限）各级的值。runner 每 tick 把被改动的 (关节, 原因, 原始值, 各级值, 最终值, 限位, 力矩裁前/上限) 写入 `safety_clips.csv`，连续 tick 合并为“裁剪段”（持续时间 = 段内 tick 数 × 20 ms）写入 `run_report.json.safety_clips`，并在 `anomaly_summary.txt` 的 `SAFETY_CLIP` 中点名关节。用于后续判断哪些门真正在改变 Policy 指令（`analysis/pm_analyze.py <run> safety`）。
