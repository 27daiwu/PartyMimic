# LOGGING.md — P5.1 日志系统（每次测试自动生成，格式固定，可直接用于离线分析）

目标：真机每跑一次，就能在开发机上 ① 逐 tick 复算 Policy，② 重建 Sim2Sim 初态，③ 分析动作跟踪误差、PD 增益、延迟、安全门和退出机制，不需要再回现场补数据。
所有时间戳 = `mono()`（机载电脑 steady_clock，秒）；`manifest.json` 的 `wall_minus_mono` 可换算墙钟。

## 1. 一次运行的目录 `logs/runs/<时间>/`

| 文件 | 内容 | 写入方式 |
|---|---|---|
| `manifest.json` | 程序版本/编译时间、**可执行文件、policy.onnx、policy_params、motion.bin、fk_chain、site.conf、modes.conf 的 sha256**，与冻结值是否一致，运行参数（模式、run_s、authority、ramp、ref_mode、命令行），kp/kd/default/action_scale/限位/力矩限，安全门阈值，obs 布局 | CHECKING 阶段写一次 |
| `trace.csv` | **每个控制 tick 一行**（见 §2） | 异步 |
| `obs.bin` | 每个 tick 送入 ONNX 的**原始 4 个输入**（float32，逐位）→ 逐 tick 回放 | 异步 |
| `state_stream.bin` | **每个 rt/lowstate 样本**（真机实测约 1 kHz，G1 `tick` 是毫秒计数；dev 回环 500 Hz）：接管前 5 s（pre-roll）+ 运行 + 退出确认后 2 s | 异步 |
| `exit_frames.csv` | 不在 trace 里的下发帧：接管握手帧 `ACQUIRE`、退出期 `HOLD`/`DAMP`（100 Hz），含实测 q 与下发 q/kp/kd | 异步 |
| `safety_clips.csv` | **每个被裁剪的 (tick, 关节) 一行**：原因、原始值、各级裁剪后的值、最终值、关节限位、速度步长、力矩（裁前/上限）、实测 q | 异步 |
| `events.jsonl` | 时间线（见 §3）：阶段、RPC 请求+响应、FSM 变化、遥控器事件、退出各步、policy_reset、log_stats | 异步 |
| `service_events.jsonl` | 服务侧：按键、状态机提示、每句语音、runner 阶段 | 同步小文件 |
| `run_report.json` | 结果、退出原因、故障、退出时序、哈希、接管前/接管瞬间状态、**timing / safety_clips / logging / timeline / anomaly** | 结束时 |
| `anomaly_summary.{txt,json}` | 异常摘要（中文），`level` = OK/WARN/FAIL | 结束时 |
| `ready.json` `admission.json` `console.log` `runner.out` | 准入明细、控制台 | |
| `guardian_report.json` `guardian_takeover/` | runner 崩溃时 guardian 接管的结果与它自己的日志目录 | 服务补充 |

guardian 接管时另写 `logs/guardian/takeover_<时间>/`（events、每个 HOLD/DAMP 帧、接管后 lowstate 流、report、anomaly_summary）。

## 2. trace.csv 列
每行 = 一个控制 tick（50 Hz）。`q:*` 等为 29 个关节各一列（`q:<joint_name>`）。
- 兼容旧列：`t, phase, ssb_frame, authority, infer_ms, step_ms, state_age_ms, tick, keys_b2, keys_b3, quat_*, gyro_*, roll, pitch, yaw`
- **观测侧实测**：`q, dq, tau_est`（lowstate）、`quat_*`（IMU 姿态 wxyz）、`gyro_*`、`rpy_*`、`tilt`、遥控器字节 `keys_b2/b3`
- **Policy**：`action`（原始输出，float32）、`action_prev`（本 tick 观测里的上一动作 = 历史 action）、`q_policy = default + action×scale`
- **参考动作（step 0）**：`ref_jp:*`、`ref_jv:*`、`ref_rp_*`、`ref_rq_*`、`ref_frame`、`ssb_frame`
- **PD 目标与最终指令**：`q_pd = q_hold + authority×(q_policy − q_hold)`（进入安全门前）、`q_cmd`（安全门后实际下发，float32）、`kp:*`、`kd:*`、`tau_pd = kp(q_cmd−q) − kd·dq`；未下发的 tick `q_cmd=nan`
- **时序**：`n`（tick 序号）、`t_mono`、`dt_ms`（控制周期）、`infer_ms`、`step_ms`、`sync_wait_ms`、`state_age_ms`（取状态时年龄）、`state_age_send_ms`（下发时年龄）、`send_us`（Write 耗时）、`dds_loop_ms`（本机 DDS 指令回环延迟）、`seq`/`seq_delta`（lowstate 样本号/本 tick 新增样本数，0=重复使用旧状态）、`tick`/`tick_delta`（机器人侧 tick 计数）、`ls_rx`/`ls_lost`（lowstate 接收总数/估计丢帧）、`ls_gap_ms`、`flags`
- `flags` 位：1 控制周期超时(>30 ms) · 2 重复状态 · 4 等新状态超时 · 8 有裁剪 · 16 lowstate 丢帧 · 32 推理 >15 ms
- 裁剪摘要：`clip_n`（本 tick 被裁关节数）、`clip_mask`

## 3. events.jsonl 的 kind
`phase`、`manifest`、`pre_takeover_state`、`preflight`、`lock_acquired`、`entry_checked`、**`rpc`**（api / arg / rc / fsm / t_req / t_resp / latency_ms / exception：GetFsmId、SwitchToUserCtrl、SwitchToInternalCtrl 的每一次请求和响应）、**`fsm_change`**（from→to）、`SwitchToUserCtrl`、`user_control_confirmed`、**`policy_reset`**（q、dq、四元数、gyro、yaw 偏移、q_hold、SSB 帧：Sim2Sim 初态）、**`remote_keys`**（每次按键变化：键名、字节、是否已接管）、`remote_start_double_click`、`remote_l2b_emergency`、`EXIT_TRIGGER`、**`exit_submit`**（决策、姿态）、`exit_rpc_attempt`/`exit_rpc_result`、`exit_fsm_matched`、`exit_frames_stopped`、`exit_stable_confirmed`、`exit_escalate_passive`、`exit_result`、`exit_outcome`、`released`、`state_stream_begin/end`、`post_exit_end`、`CRASH`（致命信号）、`cmd_rejected`、`log_stats`（丢弃计数、写线程最大滞后、lowstate 间隔）。

## 4. 异步写入与异常保全
- 控制线程只把一个结构体拷进预分配的有界队列（实测 <20 µs），**不等磁盘**；写线程用 `write(2)` 直写（无 stdio 缓冲）、每秒 `fdatasync`。回归 S29：写线程卡 1.5 s，控制周期 p99 仍 <22 ms，0 丢记录。
- 队列满只丢并**计数**（`log_stats` / `run_report.logging.dropped`，异常摘要 `LOG_DROPPED`=FAIL；**V1.6：只统计流式阶段的真实丢弃**，pre-roll 环滚动淘汰另计 `state_ring_evicted`，见 §8）。准入检查 `log_disk_space`（<100 MB 拒绝）和 `run_log_open`。
- 崩溃：SIGSEGV/ABRT/BUS/FPE/ILL 与 `std::terminate` 写 `CRASH` 事件并排空队列（2 s 内强制结束）；`fast_exit` 也排空；`kill -9` 最多丢最后约 10 ms。runner 无报告结束时，服务自动 `partymimic logtool finalize` 补写 `run_report.json`（`result=CRASH_NO_REPORT`，`reconstructed_from_logs=true`）和异常摘要；guardian 的接管结果另存。
- 接管前 5 s / 退出后 2 s 的 lowstate 全部在 `state_stream.bin`，用于分析控制权切换瞬态。（guardian 自己的日志没有接管前数据，用崩溃 runner 的 `state_stream.bin`。）

## 5. 离线工具
- 机载（C++，无需 Python）：`build/partymimic logtool check <run_dir>`（完整性）、`logtool replay <run_dir>`（obs.bin → ONNX 逐 tick 复算并与记录的 action 比对）、`logtool finalize <run_dir>`
- 开发机（`analysis/`，numpy + onnxruntime；**不在机器人包里**）：`python3 analysis/pm_analyze.py <run_dir> all` →

| 子命令 | 用途（下一轮优化） |
|---|---|
| `replay` | 逐 tick Policy 回放（Python onnxruntime），obs 与 trace 一致性 |
| `init` | Sim2Sim 初态：`initial_state.json/.npz`（q、dq、quat、gyro、yaw 偏移、q_hold、增益、限位、接管前 lowstate 窗口） |
| `tracking` | 动作跟踪误差：关节 vs 参考、vs 指令、根姿态 vs 参考根（yaw 对齐）、误差随时间 |
| `pd` | PD 增益：实际 kp/kd 回归、PD 力矩 vs 实测力矩的相关与滞后、力矩余量 |
| `latency` | 控制周期、推理、状态年龄、Write、DDS 回环、lowstate 间隔/丢帧/传输抖动、指令→响应延迟、RPC 延迟 |
| `safety` | 安全门：各关节/原因的裁剪段、门对指令的改动量、实测对限位/速度/力矩的余量 |
| `exit` | 退出机制：detect→RPC→FSM→稳定时间线、各步姿态、接管/退出前后状态窗口、交还关节跳变 |

`analysis/test_pm_analyze.py <run_dir>` 自检（回放逐位一致、恒等式 `q_pd/q_policy/tau_pd`、各分析可计算）。

## 6. 说明与局限
- “DDS 延迟”：`dds_loop_ms` = 本机发布 → 自己的订阅者收到的时间（DDS 栈 + 调度），`send_us` = Write 调用耗时，`state_age_*` = 状态样本到达到被使用的年龄，lowstate 传输抖动由「到达时间 − 机器人 tick」去漂移得到。**机器人侧收到指令的真实单向延迟无法由机载程序测得**，只能用 `command_to_response_lag`（指令变化到实测变化的相关滞后）间接估计。
- `tau_est` 是驱动器估计力矩（含摩擦/负载），不等于 `tau_pd`；分析时用 `pd` 子命令的回归与滞后。
- 每次 12 s 运行约 10 MB（≈0.9 MB/s，线性增长：60 s 约 55 MB；runner 的 `log_disk_space` 准入项会按本次 run_s 显示估算）（trace 3 MB + obs 3.5 MB + state_stream 4 MB）；`site.conf` 的 `log_obs 0` / `log_state_stream 0` 可关闭二进制流，**不建议**。

## 7. P5.3 新增
- `trace.csv`：新增 `tau_ff:*`（实际发出的前馈力矩，默认 0）、`tau_train:*`（训练执行器对**未裁剪** `q_policy` 会给的力矩，力矩上限内饱和）。
- `run_report.json`：
  - `tracking`：**分开统计** Policy 原始目标 `q_policy`、裁剪后指令 `q_cmd`、实际位置 `q`、参考 `ref_jp` 的 rms/max（`policy_vs_reference`、`command_vs_reference`、`actual_vs_reference`、`actual_vs_command`、`guard_removed_policy_minus_command`）；`clip`（有裁剪的 tick 比例 vs **原始目标越出硬限位的 tick 比例**）；`per_joint`；`torque_vs_training`（与训练执行器相比的力矩缺口）。
  - `attitude_extremes`：最大倾角/角速度 = max(trace 各 tick，**接管到退出结束之间的全部 lowstate 样本**，**触发退出的那一帧**)。触发帧不是 trace 行（循环在记录前就退出），所以只看 trace 会低估（P5.2 run3：0.26 → 实际 0.51）。`guard_stats.max_tilt` 同步包含触发样本。
  - `manifest.json`：`lead_in_frames`、`guard_torque_ff`。
- `events.jsonl`：`policy_state_reset`（运行开始时 Policy 状态校验：参考帧索引 0、上一动作 0、历史全相同、`clean`）；服务侧 `service_events.jsonl`：`run_state_cleared`。
- `anomaly_summary`：`SAFETY_CLIP` 给出逐关节明细；新增 `RAW_TARGET_OUTSIDE_LIMITS`（越限比例是 Policy 输出的性质）、`TORQUE_SHORTFALL_VS_TRAINING`；`TILT` 用上面的全量最大值。
- 回放容差：同一 CPU 架构逐位一致；机载（aarch64）日志在 x86 上回放有 ≤1e-5 的 float32 舍入差。

## 8. V1.6：日志丢弃口径、dance 时间轴、分段统计（依据 logs(7) 三次真机运行）
**(a) `dropped_states` 误报的根因与修复。** 第一次真机运行 `log_stats.dropped_states=48784` → `LOG_DROPPED` FAIL，后两次为 0。核对：events.jsonl 的 `standby_trigger`（F1+F3）到 `state_stream_begin`（L1+A 后接管）：第一次相隔 56.6 s（READY 后等了 56 s 才按 L1+A；`standby_s` 1117 s 是服务空等），后两次 4.2 s / 3.9 s。ring 容量 16384、lowstate 实测 1040 Hz → 环约 15.8 s 即满，满后每个新样本覆盖最旧一个；覆盖次数 ≈ (56.6 s + 回灌的 standby 私有环 ~6.2 s) × 1040 − 16384 ≈ 48.9 k，与 `dropped_states` 48784 吻合（后两次相隔 <5 s，环没满 → 0）。`state_stream.bin` 里 pre-roll 恰好 5.0 s，且序号连续（仅 3 个 1–3 样本的缝），**没有丢日志**。旧代码在 `BQ::push(drop_oldest)` 里无论是否 drop_oldest 都 `dropped++`，把设计内的“滚动窗口淘汰”算成了丢弃。
- 修复：`log_stats`（`log_stats_ver:2`）字段 —— `dropped_*`（**真实**丢弃，>0 即日志缺陷）；`state_ring_evicted`（pre-roll 环滚动覆盖，设计内）；`state_preroll_trimmed`（begin_stream 时早于窗口被裁掉）；`state_pushed / states_written`（恒等式 `pushed = written + dropped + ring_evicted + preroll_trimmed`，压测校验）；`preroll_written_s`（实际写出的 pre-roll 秒数）；队列峰值 `max_*_queue` 与容量 `*_cap`；`writer_pass_max_ms`、`fdatasync_max_ms/mean_ms`；`lowstate_dup_ticks`、`lowstate_nominal_dtick`。
- 另修两处真实缺陷：① writer 每轮上限（事件 4096 / tick 512 / state 4096 / frame 1024 / clip 4096）原写法 `while (pop(x) && guard++ < N)` 先取后判，达到上限时**静默丢掉已取出的那一条**；改为先判后取。② `begin_stream()` 时 ring 通常已满，流刚开始就满队列，接管后头几个样本被丢（真机日志 4 个，seq 缺口正在 ARM 时刻）；现在 begin_stream 先把早于 pre-roll 窗口的样本裁掉并留出余量（裁掉的最旧样本计入 `state_preroll_trimmed`，窗口变短会体现在 `preroll_written_s`）。
- 判定：`logtool` 的 `LOG_DROPPED` 只看真实丢弃（旧版日志用 `state_stream.bin` 序号对账重新分类：真实丢失 4 条 → WARN，其余 48 780 条 → INFO `LOG_PREROLL_ROLLING`）。
**(b) 高频通道审计（V1.6 压测 `dev/log_stress`，1 kHz / 2 kHz，慢盘）。** state 队列容量 16384（≈16 s@1 kHz，≈8 s@2 kHz）；流式阶段 writer 每轮 4096 条、每 4 ms 一轮 → 理论吞吐 ≈1 M 条/s，远高于 1 kHz；实测流式阶段队列峰值 <50（2 kHz），`fdatasync` 约 12–25 ms/次（每秒一次，只阻塞 writer 线程，控制线程不等盘；writer 卡 2 s 队列峰值 ≈2900 仍零丢失）。tick/obs（50 Hz，容量 1024 = 20 s）、events（8192）、frames/clips 均无积压。standby 私有 `pre_ring` 保存 `log_preroll_s + 1` 秒，接管时回灌进 run log 的 ring，窗口长度正确（实测 5.0 s）。
**(c) `lowstate_lost_est` / “tick 跳变”口径。** lowstate 约 1 kHz，`tick` 为毫秒计数；监视器按**首 16 个间隔的中位数**取名义步长（1 kHz → 1，500 Hz → 2），缺 1 个 tick = 缺 1 ms 样本，**不是漏了一个 20 ms 控制周期**，也**不把 500 Hz/1 kHz 间隔误判为丢包**；`ls_lost` 是自 F1+F3 触发累计（含 READY 等待），控制窗口内的增量才是有意义的，logtool 现在给出“控制窗口内缺失 tick”“控制窗口最大到达间隔”和重复 tick（同 tick 的重复样本不算丢失）。真机三次：窗口内缺 16 / 23 / 14 个 1 ms tick，最大到达间隔 10.6 / 11.9 / 12.6 ms（< 20 ms 控制周期）→ INFO，不再 WARN。
**(d) trace 时间轴（SSB 首帧等遗留语义）。** `ssb_frame` 其实是**含前导冻结**的回放索引（`lead_in_s`=1 s=50 帧的冻结首帧 + 舞蹈）。V1.6 新增列（追加在行尾，旧列/旧读者不受影响）：`dance_frame = ssb_frame − lead_in_frames`（前导期为负，0 = 舞蹈第 0 帧即动作首帧）、`dance_t = dance_frame / fps`、`seg`（0 前导 / 1 舞蹈 / 2 末帧保持 / 3 暂停参考=hold 模式）。旧日志由 `pmlog.RunLog` 按 manifest 的 `lead_in_frames / motion_frames / ref_mode` 推导同名列。报告里原先的“第 370 帧”= `ssb_frame` 370 = 舞蹈第 320 帧（6.4 s）。`n_ticks`（trace 行数）含前导和末帧保持；舞蹈 tick 数用 `RunLog.n_ticks_dance` / `tracking_segments.dance.ticks`。
**(e) `run_report.tracking_segments`**：`lead_in / dance / end_hold / paused_reference / takeover_first_tick`，各段 tick 数、关节 vs 参考 rms/max、vs 指令 rms、最大倾角、参考倾角最大值、重力方向偏差（最大/均值）、倾角−参考倾角。接管首 tick（authority 斜坡起点，|q−ref| 最大）单列，不并入任何段。`tracking`（总体）仍保留旧口径（全部 tick 混合）并注明；`pm_analyze.py tracking` 同样分段，`error_vs_time` 带 `dance_t / seg`，根姿态误差分段。yaw 对齐用 `policy_reset.yaw_offset`（SSB 首帧）：前导期参考恒为该首帧（实测 `ref_rq` 前导恒定且等于舞蹈第 0 帧），所以前导与舞蹈之间没有 yaw 偏置，无需修改。
**(f) 参考感知的 `TILT`（日志摘要侧；运行时判定由 A1 实现）。** 报告：实测最大倾角、同一时刻的参考倾角、重力方向偏差（yaw 无关）、有效参考样本内的最大偏差和倾角−参考。FAIL 仅当：偏差 > `ref_dev_rad` 且倾角 > `ref_dev_min_tilt_rad`；或倾角 > `unstable_hard_tilt_rad` 绝对硬上限；或无有效参考阶段（前导/末帧保持/暂停/旧日志）倾角 > 固定 `unstable_tilt_rad`。阈值读自 `events.jsonl tilt_judge_config`（V1.6 runner 写入）；旧日志用 A1 默认值（0.70 / 0.35 / 0.25 / 0.40）。真机三次日志重判：全部由 FAIL 变为 INFO（倾角 0.42 与参考 0.40 一致，偏差 0.02 rad）。

