# PartyMimic G1 真机部署包（纯 C++17 轻量版，P4.1：Policy 稳定性 + 退出机制重构 + 启动参数修复）

> **P4.1 修复（部署 Bug）**：P4 包里 `start.sh --accept-unvalidated` 经 service 把该参数原样转给 guardian，而 guardian 的解析器把它当成“需要取值的选项”→ `guardian: missing value for --accept-unvalidated`（rc 2），service 每秒重启一次共 178 次，F1+F3 因“guardian 不在线”被拒并播报“检查失败”。
> 修复：① 各子命令的“无取值开关”集中定义在 `app.hpp`（`flags_service/runner/guardian`），解析器与命令行构造共用同一份；② service 不再把 runner 专用选项转给 guardian（`guardian_cmdline()` / `runner_cmdline()` 统一构造）；③ guardian 以 rc 2（参数/配置错误）退出时**立即锁定、停止重启循环**，其它快速失败按 1/2/4/8/16 s 退避、5 次后锁定；原因（guardian 日志最后一行）写入 `service.log`、`state.json`（`guardian_fatal`/`guardian_why`）、F1+F3 拒绝提示与 `start.sh` 输出；④ `start.sh` 启动后**真实验证** guardian 存活（15 s 内 `guardian_alive:true`），否则打印原因并以 1 退出；⑤ selfcheck `7e` 检查 service 构造的所有 guardian/runner 命令行能被对应子命令解析；回归 S24（经 `start.sh` 真实入口，guardian 8 s 持续存活且只启动一次、F1+F3→READY、`stop.sh`）与 S25（guardian 启动失败路径）。
> **未改动**：Policy、控制律、安全阈值、WalkRun 退出策略、放行门（`--accept-unvalidated` 仍需每次键入确认）。

> **P4 与上一版（真机五次测试，`deploy/dist/partymimic_cpp.zip`，已冻结备份于 `results/p4/baseline_frozen/`）的核心差别**
> 1. **控制律**：旧版 `q_cmd = q_hold + 0.3·(q_target − q_hold)`（30% authority）是**不稳定区**（Sim2Sim 复现两次倒地；a=0 纯保持 1.0–1.7 s 内倒，a=0.3 在 1.2–2.0 s 倒）。新默认 authority=1.0，按 `ramp_s=0.3 s` 平滑上升；持续 authority∈(0.05,0.9) 被拒绝（`--allow-partial-authority` 才放行）。
> 2. **Policy**：`assets/policy.onnx` = P4 鲁棒候选（robust_d it12000，P2 同配方继续训练 + 初态/增益/延迟随机化）；冻结的 P2 it1500 基线哈希保留在 `frozen.hpp` 的 `P2_BASELINE_SHA256_POLICY`。参考动作 `motion.bin`、`policy_params.txt` 不变。
> 3. **退出**：统一 ExitPolicy（WalkRun 优先，立即请求；FSM 确认；Passive 兜底），见 `EXIT_POLICY.md`；**RPC 成功但 FSM 不匹配不再报 COMPLETED**。
> 4. **安全门**：审计放宽见 `SAFETY_AUDIT.md`。

```bash
unzip partymimic_cpp_p4.1.zip && cd partymimic_cpp
./build.sh      # 在机载电脑(aarch64)上编译，约 1–3 分钟；不连接机器人，结束时自动做离线自检
./start.sh      # 启动遥控器监听服务（会让你确认防护；不会自动动机器人）
```

纯 C++17：Unitree SDK2 C++ + ONNX Runtime ARM64 + Eigen + CMake。**没有 Python / PyTorch / MuJoCo / 机器人 XML·STL 模型**。
优先复用机器人已有依赖：Unitree SDK2 与 CycloneDDS（`/usr/local`，与 `wbc_nx/remote_start` 相同）、Eigen（`/usr/include/eigen3`）；
ONNX Runtime 若系统已装则用系统的，否则用包内 `third_party/onnxruntime`（官方 aarch64 CPU 版 1.22.0，仅头文件 + `.so`）。

## 遥控器与语音（机器人先用原厂遥控器站在 WalkRun）
键位、手柄解码、语音（G1 `AudioClient::TtsMaker` / `SetVolume`）、进程启动方式均沿用 `wbc_nx/remote_start`（`include/gamepad.hpp` 原样复制）。

| 操作 | 作用 | 语音 |
|---|---|---|
| `./start.sh` | 开始监听 | “服务已启动” |
| **F1+F3** | 加载 Policy + 实时安全检查 → `READY` 或 `BLOCKED`（此阶段机器人不动） | “正在加载” → “准备就绪” / “检查失败” |
| **L1+A** | （READY 后）等所有按键松开 → 复核 → 从 WalkRun 接管 → 执行 → 自动交还 WalkRun | “开始执行” → “执行完成” |
| **双击 START** | 任何阶段中断（接管前：什么都不发；接管后：**立即**请求 WalkRun 并确认，已倒地→PASSIVE） | “已退出”（WalkRun 确认后）|
| **L2+B**（原厂阻尼组合键，持续 ≥0.1 s） | **仅运行期有效**的紧急 PASSIVE（阻尼帧 + `SwitchToInternalCtrl(PASSIVE)`，不经 WalkRun）；接管前按下无效；不影响 F1+F3 / L1+A / 双击 START | “紧急被动” |
| 故障（推理异常/NaN/超时/看门狗/关节超速…）或失稳 | **立即**请求 WalkRun，FSM 确认 + 稳定窗口后报告；服务进入 HOLD（`./ack.sh`） | “异常已交还” |
| WalkRun 恢复失败（RPC 成功但 FSM 不在 500/801/802，或随后掉到 FSM 1）/ 已倒地 | 最终 PASSIVE，`run_report.json` 记 `WALKRUN_RECOVERY_FAILED` / `FALLEN_PASSIVE`；HOLD | “恢复失败” |
| 退出未确认 / runner 崩溃（guardian 同策略接管） | 继续阻尼，HOLD；guardian 的结果决定播报 | “检查失败” / 同上 |
| `./stop.sh` | 停服务（运行中则立即交还 WalkRun） | “已退出” |
| `./stop.sh --estop`（`/tmp/partymimic_cpp_ESTOP`） | 独立紧急通道：直接 PASSIVE | “紧急被动” |

语音只有 10 句（原 7 句 + “异常已交还 / 恢复失败 / 紧急被动”，各 ≤6 字），按**真实结果**播报（`run_report.json` 的 `voice`，在 RPC 与 FSM 确认**之后**发）；详细原因只写 `logs/`。语音在独立线程里异步执行、有界队列，永远不会阻塞控制线程；音频服务不可用时只写日志。`site.conf` 里 `tts 0` 或 `./start.sh --no-tts` 关闭。
F3 即 SDK/`remote_start` 中的 `F2` 位（`wireless_remote` 字节 2 的 bit7）。按键无反应时看 `logs/keys.log`。
`./start.sh` 提示时输入 `YES` 才会“武装”（防跌倒保护到位、遥控器急停在手、≥2 人）；否则只做检查，L1+A 无效。
可选参数：`--stage hold|clip|dance --iface eth0 --entry-fsm N --run-s S --ref running|paused --authority 1.0 --ramp-s 0.3 --end-hold-s 0.5 --foreground --no-tts`。`hold`=冻结 SSB 第 0 帧 3 s；`clip`=播放舞蹈前 `run_s`（默认 3 s）后冻结参考 0.5 s 再交还；`dance`=整段 SSB（7.1 s）+ 0.5 s。动作窗口/权限的上限是 `config/site.conf` 的 `max_run_s`（默认 12）/`max_authority`（默认 1.0），编译期绝对上限 12 s / 1.0；超过旧上限 1.0 s / 0.3 时日志、`ready.json`、`run_report.json` 仍标注 `EXCEEDS THE LEGACY CAPS`（仅提示）。

## 受保护真机验证流程（P4；每步看 `logs/runs/<时间>/run_report.json` 与 `trace.csv`，**离线验证已完成，真机尚未执行**）
前提：吊带/龙门架 + ≥2 人 + 遥控器急停（L2+B）在手；机器人用原厂遥控器站稳在 WalkRun（现场固件 FSM **802**）；`./build.sh` 的 selfcheck 全 PASS。
| 阶段 | 命令 | 通过判据（`run_report.json`） |
|---|---|---|
| S0 干跑 | `./start.sh`（不输入 YES）→ F1+F3 | READY，ready.json 的 admission 无 FAIL；不动 |
| S1 首帧保持 3 s | `./start.sh --stage hold` → F1+F3 → L1+A | `COMPLETED`，`final_fsm` ∈ {500,801,802}，trace 中最大倾角 < 0.2 rad，无 `UNSTABLE` |
| S2 双击 START | S1 运行中双击 START | `EXITED_WALKRUN`（语音“已退出”），detect→rpc_sent 与 rpc 往返记录在 `exit_detail` |
| S3 舞蹈前 2 s | `./start.sh --stage clip --run-s 2.0` | 同 S1；handback 时倾角/角速度记录在 trace 末行 |
| S4 舞蹈前 4 s → 整段 | `--stage clip --run-s 4.0` → `--stage dance` | 同 S1；任一次 `WALKRUN_RECOVERY_FAILED` / `FALLEN_PASSIVE` 即停止并回到离线分析 |
S1 之前不要直接做 S3/S4。每步结果与 Sim2Sim 预期的对照见 `results/p4/P4_REPORT.md`。

## 安全准入（F1+F3 与 L1+A 复核；FAIL 阻断，WARN 只记日志）
**FAIL（阻断）——只保留安全相关项：** 通信中断（lowstate 无数据 / <15 Hz / tick 不走）· 状态越过硬限位或护栏 · 关节不在硬限位内 · 入场倾角 ≥ min(0.5, 失稳阈值 0.40) rad（跌倒保护）· 入场 FSM 不在 `entry_fsm_allowed`（0/1 恒拒绝）或 `GetFsmId` 失败 · mode_machine 不符 ·
**有源竞争控制器**（`rt/user_lowcmd` 或 `rt/arm_sdk` 窗口内实际收到样本）· 控制权锁/急停锁占用 · guardian 不在线 · 冻结文件 sha256 不符 · L1+A 时 FSM 与 READY 时不同（原厂遥控对按键有反应或模式被改）。
**WARN（不阻断）：** lowstate 15–30 Hz · tick 推进慢 · 机器人未完全静止（0.5–2 rad/s）· 推理耗时 >15 ms · `rt/lowcmd` 未设基线 / 无流量。运行中仍有 >60 ms 控制步超时、通信超时等保护。

**FSM 802 的语义**：取自机器人上的 `wbc_nx`（`main.cpp`/`replay.cpp`）——801（kWalkingMode）与 802（kWalkingMode1）是它管理的两个官方行走模式，505 是遥操模式，退出时按进入前的 801/802 恢复。因此 802 按“官方 WalkRun 家族”直接接受（`entry_fsm_allowed 500 505 801 802`），交还时 500/801/802 都算 WalkRun。
本程序**不会**在接管前自己切 FSM：FSM 不是上述 id 时 F1+F3 拒绝并提示先用原厂遥控进入 WalkRun。
**DDS Writer 不再按数量判定**（现场日志：`rt/lowcmd` 2 个、`rt/arm_sdk` 2 个、`rt/user_lowcmd` 3 个，全是原厂服务的空闲 writer + 本程序 guardian，且无流量）。分类：guardian（自己的预建 writer，从心跳文件读数量）/ 原厂 `rt/lowcmd` 运控服务（有流量）/ 原厂空闲 writer（无流量，只记录）/ **有源竞争控制器（有流量，阻断）**。`rt/lowcmd` 数量仅在 `site.conf` 设了基线（`oem_lowcmd_confirmed 1` + `oem_lowcmd_expected_writers N`）时才作为变化检测。每次的 writer、流量、分类写在 `ready.json` 与 `console.log`。
**与原厂遥控功能无冲突**：现场日志中 3 次 START 双击和 L1+A 之后 FSM 仍是 802（原厂在 802 下对这些键无动作）；程序仍做运行时确认——L1+A 只在所有按键松开 0.25 s 后才放行（3 s 内未松开则取消），接管前再读一次 FSM，与 READY 时不一致就拒绝。

## 与 Python 栈的数值一致性（P4：Policy = robust_d；观测/动作/映射/PD 链路与 P2 完全相同）
Observation（10 步 proprio 历史、yaw 对齐的 root 四元数历史、12 步未来参考关节/根姿态）、归一化（float32 输入、重力投影）、动作映射
`q_target = default + action × scale`、PD 增益、50 Hz 控制频率，均从原 Python 栈（sim2real `Tracking` + `supermimic` observations）逐项移植。
参考动作在开发机预处理为 `assets/motion.bin`，yaml/机器人限位展开为 `assets/policy_params.txt`，两者与 ONNX 的 sha256 编进程序，每次 F1+F3 都校验。
开发机验证：354 帧金标准状态回放（对冻结 P2 基线 ONNX，`dev/parity_test <pkg> <baseline.onnx>`），**action 与 q_target 与 Python 栈最大差 0.0（逐位一致）**；P4 候选在真机状态回放（5 次真机 trace）上 Python 复刻与 C++ 记录的 action 最大差 ≤0.01（不含 reset 状态差异）；`./build.sh` 末尾的 `selfcheck` 在机器人上再比对前 3 帧并测推理耗时（开发机 p99≈1.2 ms，控制周期 20 ms，看门狗 100 ms）。
说明：参考动作在 Python 栈里是 float16，yaw 投影按 float16 逐步舍入——C++ 复刻了这一点，否则 action 会偏 ~3e-2。
实时 FK：冻结 yaml 的锚点是 `pelvis`（浮动基座），FK 退化为归一化的根四元数；`assets/fk_chain.txt` 是离线导出的 root→anchor 运动链，`FkChain` 对任意锚点通用（开发机对 MuJoCo 的 5 个 body×200 组随机姿态误差 <1e-15）。

## 安全机制（始终在线，代码强制）
控制权互斥（独占 flock + `rt/user_lowcmd` 外部写者检测 + 急停锁）· 关节硬限位（入场零容差 + 运行中护栏）· 通信超时/卡死（lowstate>60 ms、控制环看门狗 100 ms）·
指令滤波（位置限幅 margin 0.005 rad、目标限速 1.0×硬件速度、PD 力矩 1.0×硬件力矩、kp 不超过 yaml；审计见 `SAFETY_AUDIT.md`）· 独立 guardian 进程（runner 崩溃/挂起/START·L2+B 未响应时接管）·
**两级姿态保护**：失稳（倾角>0.40，或倾角>0.30 且 |陀螺 xy|>8 rad/s / 倾角变化率>3 rad/s；P4 Sim2Sim 标定）→ **立即**请求 WalkRun；倒地（>1.0 rad）→ PASSIVE。检测在 3 ms 线程与控制环两处，全部阈值可由 `site.conf` 覆盖。
**统一退出策略**（`src/exit_policy.cpp`，runner / guardian / release 共用，详见 `EXIT_POLICY.md`）：触发后同一线程内立即发 `SwitchToInternalCtrl`（专用 RPC 线程，不等阻尼/姿态），RPC 等待期间持续发“保持帧”（触发时刻 q 快照 + 保持增益，经 guard 限幅；PASSIVE 路径才发阻尼帧）；
**确认 = `GetFsmId` 返回码 0 且 FSM 在请求模式的预期集合（WalkRun 族 {500,801,802} / PASSIVE=1），并在 dwell 0.3 s + 稳定窗口 0.8 s 内持续成立**，RPC 返回成功但 FSM 不符一律不是 COMPLETED。
结果：`COMPLETED` / `EXITED_WALKRUN` / `WALKRUN_RECOVERY_FAILED` / `FALLEN_PASSIVE` / `EMERGENCY_PASSIVE` / `HANDBACK_UNCONFIRMED`，退出码 0 / 10 / 11 / 13 / 12 / 5。
紧急通道：ESTOP 文件与 L2+B 直接 PASSIVE。guardian 与 runner 共用同一策略，锁文件记录 `exit_phase/exit_owner`，runner 正在退出（未超时 10 s）时 guardian 不并发发起第二次退出。服务进程**只读**（不订阅命令、不调用运动 RPC，只有 runner 与 guardian 能发指令）。
不绕过：入场 fsm 必须在官方行走家族（0/1 恒拒绝）、q 在硬限位内、无有源竞争控制器、控制权互斥；准入分级见上。

## 退出结果与退出码（`run_report.json` 的 `result`；`final_fsm` / `requested_mode` / `final_mode` / `recovery_ok` / `exit_detail.timing_ms_from_detect`）
| result | 含义 | 退出码 | 语音 | 服务 |
|---|---|---|---|---|
| `COMPLETED` | 正常结束，WalkRun 族 FSM 确认且稳定 | 0 | 执行完成 | IDLE |
| `EXITED_WALKRUN` | 异常/中断（含双击 START、SIGTERM）后 WalkRun 恢复确认 | 10 | 异常已交还（操作员中断：已退出） | 双击 START → IDLE；故障 → HOLD |
| `WALKRUN_RECOVERY_FAILED` | WalkRun 请求后 FSM 不在 500/801/802 或随后掉到 1；最终 PASSIVE（FSM 仍在用户 FSM/未知时再发一次 PASSIVE） | 11 | 恢复失败 | HOLD |
| `FALLEN_PASSIVE` | 已知倒地（有效姿态 tilt>1.0）→ 直接 PASSIVE（姿态/通信不可信时先请求 WalkRun，失败再 PASSIVE） | 13 | 恢复失败 | HOLD |
| `EMERGENCY_PASSIVE` | ESTOP 文件 / L2+B → PASSIVE | 12 | 紧急被动 | HOLD |
| `HANDBACK_UNCONFIRMED` | 请求的模式（及 PASSIVE 兜底）都未确认，持续阻尼并监视 | 5 | 检查失败 | HOLD |
| `ARM_FAILED` / `BLOCKED` / `CRASH` … | 接管前/接管失败（未接管则不发任何指令） | 4 / 6 / 1 | 检查失败 | IDLE/HOLD |

## 剩余真机运动风险（状态：真机未验证）
| # | 项目 | 处理 |
|---|---|---|
| 0 | **OEM 对姿态恢复的真实能力未知**：sim 里用“倾角>0.45 时 WalkRun 落到 FSM 1”复现了两次倒地，失稳阈值 0.40/0.30 已用 Sim2Sim 标定，需用真机日志/sim 微调（`site.conf` 的 `unstable_*`） | 退出路径由 `dev/exit_paths.sh` 覆盖；阈值待标定 |
| 1 | **固件行为未实测**：`SwitchToUserCtrl` 后 fsm 是否为 1000；交还后落到哪个 fsm（`run_report.json` 的 `restored_fsm` / `restored_to_entry_fsm`）；**用户控制期间 `wireless_remote` 是否还更新**（决定运动中双击 START 是否有效，看 `trace.csv` 的 `keys_b2/b3`）；用户指令看门狗 | 首次只做 `--stage hold`（吊带保护）。若运动中 START 无效，运动期间只靠遥控器急停 + guardian |
| 2 | 原厂遥控在用户控制期间对按键的响应未知 | 同上；`logs/keys.log` 与 `trace.csv` 记录全部按键 |
| 3 | **动作范围**：P2/P4 动力学安全仍 FAIL（足底穿透 11–17 mm > 5 mm，P4 候选 17.4 mm，原生 MuJoCo 评测）；力矩在髋 roll 有最长 0.2 s 的饱和 | 按上表分阶段放大；放宽 authority 之外的上限需另行评审 |
| 4 | 入场姿态：折叠姿态（fsm 0/1）膝角超限 | 先用遥控器站到 WalkRun；检查保留，**不要改限位** |

## 日志（全部在 `logs/`）
`service.log` 总日志（含每条语音）· `keys.log` 按键 · `state.json` · `guardian/` · `runs/<时间>/`：`trace.csv`（50 Hz：关节 q/dq/力矩、IMU、Policy action/q_policy、q_cmd、推理耗时、SSB 帧、遥控字节）、
`events.jsonl`（锁、SwitchToUserCtrl、确认、`EXIT_TRIGGER`、`exit_outcome`、释放）、`ready.json`（含 run_s/authority 是否超出旧上限）、`admission.json`、`run_report.json`（`result`、`exit_reason`、`fault`、`requested_mode`、`final_mode`、`final_fsm`、`recovery_ok`、`exit_detail`：t_detect / t_rpc_sent / t_rpc_returned / t_fsm_confirmed / 稳定确认时间戳、detect→rpc_sent 延迟、FSM 轨迹）。`logs/latest` 指向最近一次。
故障后服务进入 HOLD：检查机器人和日志 → `./ack.sh`。`./stop.sh` 停服务（`--estop` 同时置软件急停）。

## 包内容（严格白名单）
```
build.sh start.sh stop.sh ack.sh CMakeLists.txt README.md EXIT_POLICY.md SAFETY_AUDIT.md
config/site.conf                    站点配置（入场 fsm、可选 rt/lowcmd 基线、默认参数、语音、run/authority 上限、退出与安全门阈值覆盖）
src/*.cpp  include/pm/*.hpp  include/gamepad.hpp
assets/policy.onnx policy_params.txt motion.bin fk_chain.txt SHA256SUMS
third_party/onnxruntime/            ONNX Runtime 1.22.0 aarch64（头文件 + libonnxruntime.so，MIT）
```
单一可执行文件 `build/partymimic`：`service`（监听+语音+监督）/`runner`（一次 F1+F3→L1+A）/`guardian`/`release`（手动归还）/`selfcheck`。
Eigen 复用系统 `/usr/include/eigen3`；若机器人缺失，可把 Eigen 头文件放进 `third_party/eigen3/` 再 `./build.sh`。
