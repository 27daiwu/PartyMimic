# EXIT_POLICY.md — 异常退出机制（单一来源：`include/pm/exit_policy.hpp` / `src/exit_policy.cpp`）

runner（`ControlSession`）、guardian（`take_over`）、`partymimic release` 三条路径**共用**同一个模块：
`decide_exit()`（决策表）→ `ExitExecutor`（定时序执行 + FSM 确认）→ `classify_exit()`（结果分类）。`handback()`（`handback.hpp`）只是 guardian/release 用的同步封装。
没有第二份退出规则。（旧的 `decide_exit_mode`、`damp_before_release_s`、"先阻尼 0.1 s 再等姿态"已删除。）

## 1. 设计原则

1. **早、立即**：官方 WalkRun 的姿态恢复能力强，异常时第一时间交还 WalkRun；不先渐变、不先阻尼、不先等姿态确认。
2. **保持而不是放松**：RPC 等待期间发“保持帧”（触发时刻测得 q 的快照 + 保持增益，经 `SafetyGuard::filter_command` 限幅），不发 kp=0 阻尼帧；只有 PASSIVE 路径才发阻尼帧。
3. **确认才算数**：`confirmed` 当且仅当 `GetFsmId` 返回码 0、FSM 属于请求模式的预期集合（WALKRUN 族 {500,801,802}；PASSIVE = {1}），并在 dwell 0.3 s + 稳定窗口 0.8 s 内**持续**成立。RPC 返回 0 但 FSM 不符 **永远不是 COMPLETED**（真机两次倒地日志里的 bug：`rpc 0, fsm 1, mode_matched=false` 被报成 COMPLETED）。
4. **独立紧急通道**：ESTOP 文件 / 遥控 L2+B 直接 PASSIVE，不经 WalkRun。
5. **单一退出所有者**：同一时刻只有一个主体（runner 或 guardian）在退出，锁文件里记 `exit_phase / exit_owner / exit_t`。

## 2. 决策表（`decide_exit`，输入 = 触发 Fault + 当前姿态）

姿态 = 最新 lowstate 的 IMU（年龄 ≤ `attitude_max_age_s`=0.10 s、四元数范数 0.9~1.1 才算可信）。tilt = 机体 z 轴与世界 z 轴夹角。

| 触发 | 条件 | 请求模式 | 类别 | 备注 |
|---|---|---|---|---|
| 正常结束（动作窗口走完） | tilt ≤ 1.0 | **WALKRUN** | NORMAL | → `COMPLETED` |
| 遥控双击 START / SIGINT·SIGTERM（stop.sh、service 停止） | tilt ≤ 1.0 | **WALKRUN** | OPERATOR | → `EXITED_WALKRUN`，语音“已退出” |
| Policy 异常：INFERENCE（含 NaN/非法输出 → CMD_INVALID）、LOOP_OVERRUN、WATCHDOG、NO_RESPONSE、CMD_INVALID、JOINT_SPEED、TORQUE_SUSTAINED、JOINT_LIMIT、GYRO、LOCK_LOST、FSM_CHANGED、SWITCH_FAILED… | tilt ≤ 1.0 且姿态可信 | **WALKRUN** | POLICY | 语音“异常已交还”（确认后） |
| **失稳** `UNSTABLE`（见 §3） | tilt ≤ 1.0 | **WALKRUN（立即）** | UNSTABLE | 3 ms 线程检测 |
| 倒地 `TILT`（tilt > 1.0 rad）/ 任何触发时 tilt > 1.0 | — | **PASSIVE** | FALLEN | → `FALLEN_PASSIVE`，语音“恢复失败” |
| 状态不可信：STATE_INVALID / STATE_FROZEN / COMM_TIMEOUT，或姿态过期/未知（**P4 修改**） | 立即 WALKRUN（厂家估计器自带状态） | WALKRUN，失败则 PASSIVE | UNTRUSTED | 已知倒地(tilt>fallen)仍直接 PASSIVE；其余先请求 WalkRun，FSM 不匹配即升级 PASSIVE |
| **ESTOP 文件** | — | **PASSIVE** | EMERGENCY | 阻尼帧 + `SwitchToInternalCtrl(PASSIVE)`，→ `EMERGENCY_PASSIVE`，“紧急被动” |
| **遥控 L2+B**（仅运行期） | — | **PASSIVE** | EMERGENCY | 同上，fault = `REMOTE_DAMP` |
| WALKRUN 恢复失败（RPC ok 但 FSM 不在 {500,801,802} / 随后掉到 1 / 仍在用户 FSM） | — | 结果 `WALKRUN_RECOVERY_FAILED`，最终 PASSIVE | — | FSM 已是 1 → 不再发 RPC；FSM 仍是用户 FSM 或未知 → 再发一次 PASSIVE 并继续阻尼 |
| 退出期间出现 ESTOP / L2+B | — | 升级为 PASSIVE | EMERGENCY | 阻尼帧立即切换；当前 RPC 返回后立刻发 PASSIVE |

重新决策：显式拒绝（3203/3204/3102）后的**一次**重试前，按当前姿态重新决策（倒地则改 PASSIVE）。超时/未知码不盲目重试；仅当 FSM 仍是用户 FSM 且 RPC 非 0 时做**一次**验证过的重试。

## 3. 失稳检测（`StabilityMonitor`，可配）

> **V1.6 A1 更新**：下面的 `tilt > unstable_tilt_rad (0.40)` 现在只在**没有有效参考**时生效；参考运行时由第 3b 节的参考感知判定取代，并新增 0.70 绝对硬上限。其余规则不变。

对每个**新的**lowstate 样本（按 `seq` 去重）评估，连续 `unstable_debounce_samples`（默认 2）个样本一致才触发：

* `FALLEN`：tilt > `fallen_tilt_rad`（1.0）→ `Fault::TILT` → PASSIVE
* `UNSTABLE`：tilt > `unstable_tilt_rad`（**0.40**），**或** tilt > `unstable_soft_tilt_rad`（**0.30**）且（|gyro_xy| > `unstable_gyro_rad_s`（**8.0**）**或** d(tilt)/dt > `unstable_tilt_rate_rad_s`（**3.0**，0=关））→ `Fault::UNSTABLE` → WALKRUN

运行位置：① `ControlSession::watch_loop`（**3 ms** 线程，只要有新样本就评估）；② runner 控制环（`check_state_once`，每 20 ms）。任一先到先触发，CAS 保证只退出一次。
入场准入的倾角门同步为 `min(max_entry_tilt_rad 0.5, unstable_tilt_rad 0.40)`，避免接管后立刻失稳。

## 3b. 参考感知失稳判定（V1.6 A1，`StabilityMonitor` + `RefChannel`）

**起因**：DIY 舞蹈 3 次真机在 ssb_frame 370–371（≈7.4 s）因固定 `unstable_tilt_rad 0.40` 退出——机器人在**正确跟随一个本身前倾 0.40 rad 的参考**（实测 tilt 0.3985/0.4012 vs 参考 0.4002，gyro 0.5–0.7 rad/s），不是失稳；整段舞蹈参考 pelvis 最大倾角 0.498 rad（>0.40 的帧 50 个、分 ≥5 段）。固定阈值与舞蹈本身冲突。

**输入**：runner 控制环每 tick 把**当前参考帧**的 pelvis 四元数（`PolicyOut.ref_rq`）和 `valid` 标志经 `ControlSession::set_reference()` 发布到 `RefChannel`（mutex，3 ms 看门狗线程与控制环两个 monitor 共用）。`valid = ref_mode==running && motion_t > lead_in_frames && 窗口未结束`（前导冻结 / paused 保持 / 末尾保持 = 无效）。参考四元数非有限/范数异常 → 无效；**年龄 > `ref_max_age_s`(0.15 s)**（控制环卡住）→ 无效。

**特征**（全部与 yaw 无关）：`g_meas`、`g_ref` = 重力方向在**各自 pelvis 系**下的单位向量；`dev = acos(g_meas·g_ref)`（重力方向误差）；`tilt`（绝对倾角）；`d(dev)/dt`（40 ms 窗口，需 ≥15 ms 历史）。

**规则**（连续 `unstable_debounce_samples`(2) 个**不同**样本一致才触发，`FALLEN` 优先）：

| 规则（`TiltRule`） | 条件 | 依赖参考？ |
|---|---|---|
| `FALLEN` | tilt > `fallen_tilt_rad` (1.0) → PASSIVE | 否 |
| **`HARD_CAP`** | tilt > `unstable_hard_tilt_rad` (**0.70**) → UNSTABLE/WALKRUN | **否**（绝对硬上限：不被参考值抬高，不依赖参考通道是否有效；参考最大倾角 0.50 + 余量） |
| **`REF_DEV`** | 参考**有效且新鲜** 且 tilt > `ref_dev_min_tilt_rad`(0.25) 且 dev > `ref_dev_rad`(**0.35**) 且 d(dev)/dt ≥ `ref_dev_growth_rad_s`(0.0) | 是 |
| `ABS_RATE` | tilt > `unstable_soft_tilt_rad`(0.30) 且（\|gyro_xy\| > 8.0 或 d(tilt)/dt > 3.0）（旧规则，保留） | 否 |
| `FIXED_LIMIT` | **无有效参考**（前导 / paused / 保持 / 缺失 / NaN / 过期）且 tilt > `unstable_tilt_rad`(**0.40**) | 否（旧的保守固定阈值） |

即：参考有效时**固定 0.40 被 REF_DEV 取代**（参考前倾 0.50 且跟随良好 → dev≈0.03–0.14 → 不触发）；参考失效时回到 0.40。`ref_aware_unstable 0` = 完全回到 V1.5.1 行为（再加 0.70 绝对上限）。

**记录（下次日志一眼可判）**：`events.jsonl` `tilt_judge_config`（开跑时的全部门限）；触发时 `tilt_judge_trigger`：`{monitor, level, rule, ref_valid, tilt_rad, ref_tilt_rad, gravity_err_rad, gravity_err_rate_rad_s, tilt_minus_ref_rad, gyro_xy, tilt_rate, ref_age_ms, limits{hard_tilt_rad, ref_dev_rad, ref_dev_min_tilt_rad, ref_dev_growth_rad_s, soft_tilt_rad, gyro, tilt_rate, fixed_limit_tilt_rad, fallen_tilt_rad, debounce_samples}}`；`EXIT_TRIGGER.detail` / `run_report.exit_reason` 文本同样含规则名、实测/参考倾角、重力误差与各门限；`run_report.json` 的 `tilt_judge`：`config` + `stats{samples, samples_ref_valid, max_tilt_rad, max_ref_tilt_rad, max_gravity_err_rad, max_tilt_minus_ref_rad, min_margin_gravity_err_rad（ref_dev_rad − dev 的最小值，tilt>floor 时）, min_margin_hard_cap_rad, min_margin_fixed_limit_rad}`（正常跑完也有 → 可量化离触发还有多远）。

**标定依据与局限**见 `results/diy/v16/A_deploy/TILT_JUDGE_REPORT.md`：整段 DIY 舞蹈 Sim2Sim（68 次含 kp0.8/延迟/摩擦/噪声/推力）最大重力误差 0.264（kp0.8，其余 ≤0.21，标称 0.10）、真机前 7.4 s 最大 0.14；261 次注入倒地 100 % 在 tilt=1.0 之前检出、提前量中位 0.36 s / p10 0.165 s / 最小 0.075 s（94.6 % ≥150 ms，57 % ≥300 ms）。**仿真≠真机**：真机重力误差约为仿真的 1.4–1.7 倍，门限 0.35 对真机最差正常样本留 ≈1.9× 余量；真机仍需现场验证，并看 `tilt_judge.stats.min_margin_gravity_err_rad`。

## 4. 状态图

```
                     L1+A, ARM ok
 IDLE/READY ───────────────────────────► NORMAL (policy drives, HOLD gains = policy gains)
 (no control; abort sends nothing)          │
                                            │ trigger_exit(fault)  [any thread, CAS once]
                    ┌───────────────────────┘
                    ▼  decide_exit(fault, attitude)
          ┌─────────────────────┬──────────────────────────┐
          │ WALKRUN             │ PASSIVE                   │
          ▼                     ▼                           │
   EXITING: RPC thread     EXITING: RPC thread              │
   SwitchToInternal(WALKRUN)  SwitchToInternal(PASSIVE)     │
   frames = HOLD(q snapshot)  frames = DAMP(kp0,kd3)        │
          │                     │                           │
          ▼ poll GetFsmId (50 ms)                           │
   fsm ∈ {500,801,802}? ──no, fsm=1 / foreign / still user──┤
          │ yes                 │                           │
   dwell 0.3 s + window 0.8 s   │  fsm == 1 → settle PASSIVE│
   (any mismatch = fail)        │  still user/unknown →     │
          │                     │  PASSIVE RPC + DAMP, poll │
          ▼                     ▼                           │
   COMPLETED / EXITED_WALKRUN   EMERGENCY_PASSIVE / FALLEN_PASSIVE / WALKRUN_RECOVERY_FAILED
                                │ not confirmed
                                ▼
                       HANDBACK_UNCONFIRMED (DAMP + keep watching 30 s)

 ESTOP file / L2+B during EXITING ─► escalate_passive(): DAMP frames at once, PASSIVE RPC after the current RPC returns
```

## 5. 时序（线程）

| 线程 | 职责 |
|---|---|
| 触发线程（watch 3 ms / 控制环 / 信号 / 遥控） | `trigger_exit`：决策（µs）→ CAS `NORMAL→EXITING` → `exec.submit()`：**① 唤醒 RPC 线程（cv）→ ② 发第一帧 HOLD/DAMP → ③ 写锁文件 `exit_phase=EXITING`** → 之后才做 guard trip / 事件日志 |
| RPC 线程（`ExitExecutor::worker`，启动时就建好） | 取到请求立即 `SwitchToInternalCtrl`（不被 100 Hz 帧流、`lk_` 锁阻塞），记录 `t_rpc_sent / t_rpc_returned`；之后轮询 FSM、确认、必要时 PASSIVE 兜底 |
| 帧线程（`ExitExecutor::frame_loop`，100 Hz） | 发送 HOLD（q 快照 + 保持增益，经 guard）或 DAMP 帧；WALKRUN 路径在观察到 FSM 离开用户 FSM 后停止写（同官方示例）；顺带检查 ESTOP/L2+B 并升级 |

`t_detect` = 决策时刻；`t_event` = 触发所依据的那个 lowstate 样本的到达时刻（失稳检测时 = 满足去抖的第 2 个样本）。run_report 的 `exit_detail`：

```
timing_ms_from_detect: {t_event, t_detect=0, t_rpc_sent, t_rpc_returned, t_fsm_confirmed, t_stable_confirmed, t_done}
detect_to_rpc_sent_ms, event_to_rpc_sent_ms, rpc_latency_ms, fsm_trace[[ms,fsm],...], rpc_codes, rpc_modes,
requested_mode, final_mode, final_fsm, recovery_ok, walkrun_recovered, confirmed, settled, escalated_to_passive, ...
```
语音与 service.log 在 RPC **之后**、按 `run_report.json` 的真实结果发（异步线程）。

## 6. 结果分类 / 退出码 / 语音 / 服务动作

| result | 条件 | exit code | 语音 | service |
|---|---|---|---|---|
| `COMPLETED` | 正常结束，WALKRUN 请求，FSM 在 WalkRun 族并稳定 | 0 | 执行完成 | IDLE |
| `EXITED_WALKRUN` | 异常/中断，WALKRUN 请求并确认 | 10 | 异常已交还（操作员中断：已退出） | 操作员双击 START → IDLE；其余 HOLD |
| `WALKRUN_RECOVERY_FAILED` | WALKRUN 未确认（FSM=1 / 异物 FSM / 仍用户 FSM 后 PASSIVE 兜底） | 11 | 恢复失败 | HOLD（`state.json` 记 `last_final_fsm`、`last_recovery_ok=false`） |
| `FALLEN_PASSIVE` | 决策即 PASSIVE（倒地/状态不可信）且确认 | 13 | 恢复失败 | HOLD |
| `EMERGENCY_PASSIVE` | ESTOP 文件 / L2+B（含退出中升级）且 PASSIVE 确认 | 12 | 紧急被动 | HOLD |
| `HANDBACK_UNCONFIRMED` | 请求模式与 PASSIVE 兜底都未确认 | 5 | 检查失败 | HOLD |
| `ARM_FAILED` 等接管前后失败 | 沿用名称；退出判定仍走同一策略（FSM 仍在入场 WalkRun → 不发 RPC，只确认） | 4 … | 检查失败 | HOLD |

`run_report.json` 新字段：`final_fsm`、`requested_mode`、`final_mode`、`recovery_ok`（= 请求的模式被确认）、`exit_class`、`voice`、`exit_detail`。guardian 报告同名字段。runner 被 kill -9 时，service 等待 guardian 的报告并按其 `result` 播报（`GUARDIAN_<result>` 记入 state.json）。

## 7. 紧急通道

* **ESTOP 文件**（`/tmp/partymimic_cpp_ESTOP`，`stop.sh --estop`）：runner 的 3 ms 线程与 `write()` 都检查 → PASSIVE。guardian 对它有 0.10 s 宽限（`--estop-grace-s`），runner 在宽限内已开始退出则 guardian 不介入。
* **L2+B**（`DampCombo`，`remote.hpp`；字节 2=0x20、字节 3=0x02，即 `0x220`）：要求 L2、B 同时按下连续 ≥ `l2b_hold_s`（0.10 s）且 ≥2 个不同样本；**仅在接管后（ACQUIRE/NORMAL）生效**；接管开始时已经按住的键必须先松开再按才算（防止 L1+A 交接时误触）；额外按键不取消（急停不应被误触的另一根手指破坏）；F1+F3 / L1+A / 双击 START 的判定器互不影响。guardian 也监听它，作为 runner 不响应时的后备（有 `exit_phase` 闸门）。

## 8. runner 与 guardian 的协同

* 锁文件（256 字节定长记录，单次 `pwrite`，读者永远看不到半写状态）：`pid / entry_fsm / exit_phase(NONE|EXITING|DONE) / exit_owner / exit_t`。guardian 的 alive 文件另带 `exit_phase/exit_owner/exit_t`（guardian 自己退出时的所有权）。
* guardian 在“runner 持锁且 `exit_phase ∈ {EXITING, DONE}` 且 `now - exit_t < exit_owner_timeout_s`(10 s)”时**不做任何触发**（包括静默检测、START/L2+B 宽限、ESTOP）；超时才接管。
* guardian 触发 → 先决策（µs）→ **SIGKILL 唯一写者并等待回收** → 同一个 `ExitExecutor`（立即 RPC + HOLD/DAMP 帧 + FSM 确认 + 兜底）。runner 已消失时（崩溃）先做一次 `GetFsmId`（`precheck`）：FSM 已是 WalkRun 家族就不再盲发 RPC，只做稳定确认。
* guardian 的心跳现在由独立线程写，接管过程中不中断。

## 9. 可配置项（`config/site.conf`，默认即下表；范围检查强制，越界 = runner/guardian 拒绝启动）

| site.conf 键 | 字段 | 默认 | 范围 |
|---|---|---|---|
| `unstable_tilt_rad` | `ExitConfig.unstable_tilt_rad` | 0.40 | [0.1, fallen) |
| `unstable_soft_tilt_rad` | `unstable_soft_tilt_rad` | 0.30 | [0.05, unstable_tilt] |
| `unstable_gyro_rad_s` | `unstable_gyro_rad_s` | 8.0 | [0.3, 10] |
| `unstable_tilt_rate_rad_s` | `unstable_tilt_rate_rad_s` | 3.0（0=关） | 0 或 [0.5, 20] |
| `unstable_debounce_samples` | 同名 | 2 | [1, 10] |
| `fallen_tilt_rad` | 同名 | 1.0 | [0.6, 1.5] |
| `ref_aware_unstable`（V1.6 A1） | `ref_aware` | 1 | 0/1（0 = 旧行为：只用固定 `unstable_tilt_rad`） |
| `unstable_hard_tilt_rad` | 同名 | **0.70** | [0.55, fallen_tilt_rad) 且 > `unstable_tilt_rad` |
| `ref_dev_rad` | 同名 | **0.35** | [0.15, 1.0] |
| `ref_dev_min_tilt_rad` | 同名 | 0.25 | [0.10, unstable_hard_tilt_rad] |
| `ref_dev_growth_rad_s` | 同名 | 0.0 | [-2, 10]（-2 = 不要求增长） |
| `ref_max_age_s` | 同名 | 0.15 | [0.04, 1.0] |
| `attitude_max_age_s` | 同名 | 0.10 | [0.02, 0.5] |
| `l2b_hold_s` | 同名 | 0.10 | [0.02, 1.0] |
| `exit_confirm_timeout_s` / `exit_fsm_poll_s` | `confirm_timeout_s` / `fsm_poll_s` | 3.0 / 0.05 | [0.5,10] / [0.01,0.5] |
| `exit_dwell_s` / `exit_stable_window_s` / `exit_min_stable_polls` | 同名 | 0.3 / 0.8 / 3 | [0.1,2] / [0.5,10] / [2,100] |
| `exit_mismatch_fail_s` | `mismatch_fail_s` | 0.3 | [0.05, 3] |
| `exit_unconfirmed_hold_s` | `unconfirmed_hold_s` | 30 | [0, 120] |
| `exit_hold_gain_scale` | `hold_gain_scale` | 1.0 | (0, 1] |
| `exit_owner_timeout_s` | 同名 | 10 | [3, 60] |
| `max_tilt_rad` 等 GuardConfig、`watchdog_s` 等 | 见 `SAFETY_AUDIT.md` | | |

## 10. 实测（loopback 模拟机器人，`dev/sim_robot` + `dev/exit_paths.sh`，x86 开发机）

**不是真机数据**：sim 里 DDS 在本机回环，RPC 本身 ~0.4–0.6 ms，真机 RPC 往返会更长；下面的数字衡量的是**本程序内部**的退出延迟。

| 场景 | detect→rpc_sent（程序内，决策→`SwitchToInternalCtrl` 调用入口） | 其它 |
|---|---|---|
| 正常结束（COMPLETED） | 0.01–0.14 ms | 稳定窗口 1.11 s 后确认 |
| 双击 START | 0.7–2.9 ms（含 DDS 回调线程 → 3 ms 看门狗线程的发现延迟，`t_detect` 取遥控回调时刻） | |
| Policy 异常：inference / nan / overrun / hang(WATCHDOG) | 0.02–0.13 ms | WALKRUN，无阻尼帧 |
| **失稳（roll 阶跃 0.30 rad，gyro 3 rad/s）** | **0.04–0.37 ms**；lowstate 样本到达→rpc_sent 1.2–1.8 ms | **端到端（sim 发布倾斜样本 → sim 收到 RPC）4–5 ms**（含 DDS、2 样本去抖、≤3 ms 轮询） |
| 倒地（roll 1.3，→PASSIVE） | 0.02–0.05 ms；样本→rpc 1.9 ms | 阻尼帧 + PASSIVE |
| ESTOP 文件 | 0.02–0.07 ms | PASSIVE |
| L2+B（0x220） | 1.2–2.5 ms（含 DDS 回调→3 ms 线程） | PASSIVE |
| guardian（kill -9 runner） | 0.4–0.6 ms（含 SIGKILL+回收 0.02–0.04 ms） | WALKRUN，无阻尼帧 |
| 慢 RPC（0.35/0.4 s） | 触发延迟不变（0.02–0.03 ms） | RPC 期间 HOLD 帧 +20..+40 帧（100 Hz）、阻尼帧 0 |

目标（≤20 ms，尽量 <10 ms）在 sim 中全部满足；真机上需要关注的是 `rpc_latency_ms` 与 lowstate 采样周期，二者在 `run_report.json` 里逐次记录。

场景通过情况：`dev/exit_paths.sh`（21 个场景，106 项检查）全部通过，包括原有 S1–S9（已按新语义更新：S1 COMPLETED 且无阻尼帧；S3 START×2 → EXITED_WALKRUN、HOLD 帧而非 kp=0；S4 guardian 同策略；S5 倒地 → FALLEN_PASSIVE）和新增：S10 失稳立即 WalkRun、S11 慢 RPC 期间继续倾斜 OEM 落到 FSM 1（复现真机两次倒地）、S12 WalkRun 恢复失败四种变体（rpc0+fsm1 / rpc0+fsm 不变 / 随后掉到 1 / 全部无效）、S14 ESTOP、S15 L2+B（含接管前按下无效、去抖）、S16 runner+guardian 同时触发不竞争（ESTOP+慢 RPC、START×2+慢 RPC、runner SIGSTOP）、S17 慢 RPC、S18 注入推理异常/NaN/overrun/hang、S19 release、S20 caps、S21 ARM 失败。

## 11. 设计取舍与已知限制（均偏保守且可配置）

1. **（P4 已改）姿态/状态不可信不再直接 PASSIVE**：60 ms 级通信抖动若直接阻尼会让正在站立的机器人倒地；现在先请求 WalkRun（用户真机经验：原厂 WalkRun 恢复力强），由 FSM 确认兜底，失败才 PASSIVE。已知倒地（有效姿态 tilt>fallen_tilt_rad）仍直接 PASSIVE。
2. **确认窗口 1.1 s 不影响交还时刻**：RPC 已发出、OEM 已接手，窗口只用于上报；服务的 run 超时是 `run_s+60 s`（runner 自身兜底 `lead_in+run_s+end_hold+1 s`；P6 起 `run_s` 可到 `ABS_MAX_RUN_S`=60 s，两个超时都随 run_s 线性放大，退出/恢复逻辑与时长无关，不受影响）。
3. **失稳阈值（P4 已用 Sim2Sim 标定，见第 10 节）**：0.40 / 0.30&(陀螺8.0 或 倾角率3.0)；**V1.6：固定 0.40 与 DIY 舞蹈本身的 0.40–0.50 倾角冲突（真机 3 次 7.4 s 误退出），参考运行时改用第 3b 节的参考感知判定**；仍需真机 `trace.csv` 复核；陀螺判据受 IMU 噪声影响，已用 2 样本去抖。
4. **WALKRUN 路径在 FSM 离开用户 FSM 后停止写帧**；若真机固件在切换窗口内仍需要持续帧，可把 `exit_hold_gain_scale` 与停写条件调整（目前未验证）。
5. 单个 WalkRun 请求若 `rpc_code != 0` 且 FSM 仍为用户 FSM → 一次验证重试；显式拒绝（3203/3204/3102）→ 立即一次重试（间隔 50 ms）。
6. 真机固件对 `SwitchToInternalCtrl(PASSIVE/WALKRUN)` 之后 FSM 的真实取值（500/801/802 的哪一个）仍未实测；确认集合取自 `profile.hpp`，需按首次真机日志核对。
7. CycloneDDS `measure_traffic` 在 GO 复核时偶发 `EntityDelegate::prevent_callbacks` 断言（1 次/约 25 次 sim 运行，**早于本次改动的既有代码**，发生在 READY 之后、接管之前，runner 崩溃 → HOLD，机器人未被触碰）。未处理，建议单独跟踪。

## 10. P4 Sim2Sim 失稳阈值标定（results/p4/scripts/thr*.py 思路，500 次滚动）
- 正常舞蹈（P4 鲁棒策略，真机初态分布，nominal）：最大 tilt p50/p100 = 0.19/0.32 rad；|gyro_xy| 峰值 p50/p100 = 3.0/5.0 rad/s；倾角变化率峰值 ≤3.0 rad/s ⇒ **基于陀螺的软判据（1.5 rad/s）在正常舞蹈中会误报**，故默认 `unstable_gyro_rad_s` 提高到 8.0（实质关闭），主判据改为倾角。
- 421 次未倒/79 次倒地的压力场景（延迟/低 kp/推力）上比较规则：`tilt>0.35` 误报 44/421、倒地检出 79/79、触发→tilt=1.0 提前量中位 0.30 s；`tilt>0.40` 误报 12/421、检出 79/79、提前量中位 0.24 s（p10 0.13 s）；`tilt>0.45` 误报 2/421、提前量中位 0.20 s。选 **0.40**：正常最大 0.32 留有余量，同时倒地前仍有 ≈0.24 s 的恢复窗口。
- 提前量很短（0.1–0.4 s）：检测→RPC 必须在毫秒级（EXIT_POLICY 实测 0.02–0.4 ms/程序内），真机上 RPC 往返和 OEM 恢复能力未知，仍需受保护真机复核。

## P5.1：退出过程的完整记录（不改变任何退出规则）
同一套 `ExitExecutor` 在 `events.jsonl` 里逐步留痕（runner 与 guardian 接管日志相同）：`EXIT_TRIGGER` → `exit_submit`（决策模式/类别/姿态）→ `exit_rpc_attempt` / `exit_rpc_result`（每次 RPC 的 mode、rc、耗时）→ `exit_frames_stopped`（FSM 离开用户 FSM，停止 HOLD 帧）→ `exit_fsm_matched` → `exit_stable_confirmed` / `exit_fsm_dropped_after_match` → `exit_escalate_passive` → `exit_result`；`run_report.json` 的 `exit_detail` 增加决策/RPC 发出/FSM 确认三个时刻的姿态（tilt、gyro_xy）与 `t_frames_stopped`；退出期每个 HOLD/DAMP 帧在 `exit_frames.csv`，退出前后的全速率 lowstate 在 `state_stream.bin`。离线用 `analysis/pm_analyze.py <run> exit` 得到时间线和交还瞬态。
