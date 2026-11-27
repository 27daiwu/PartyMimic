# PartyMimic G1 真机部署包 V1.6（纯 C++17，完整部署 + 真机数据采集）

> **V1.6（A 线 2/2：依据 logs(7) 真机三次完整舞蹈日志）**
> 1. **“开始执行”播报完成后再等 0.5 s 才接管**（`site.conf announce_wait_s`，默认 0.5，范围 0–5 s）。流程：F1+F3 检查 → L1+A（松键）→ 服务播报“开始执行”→ **播报完成** → 等 `announce_wait_s` → **重新确认接管前提** → 接管并起舞。语音服务没有“播放完成”回调（`TtsMaker` 在请求被接受后即返回，SDK 例程也是返回后 `Sleep`）：完成时刻 = `TtsMaker` 返回 + 估算语音时长（`announce_s_per_char` 0.30 s/字 + 0.3 s，“开始执行”约 1.5 s），**上限 `announce_max_s` = 4 s（从播报开始算，绝不无限等待）**；语音不可用/被拒 → 视为已完成。**等待期间机器人没有被接管（runner 处于 `GO_WAIT`：只监控，没有锁、没有 writer、不发任何 RPC）**；双击 START、L2+B、e-stop 文件（急停：此时尚未接管，所以表现为**取消且不下发任何指令**，e-stop 闩锁保持到 `./ack.sh`，下一次 F1+F3 会被 `estop_file_absent` 拦住）、服务 SIGTERM、guardian 失联、lowstate 超时(>0.25 s)、倾角/不静止、FSM 离开 WalkRun 集合、出现竞争写者、控制锁被占 → **取消**（run_report `CANCELLED_BEFORE_TAKEOVER`，rc 7，服务回 IDLE，语音“已退出”或“检查失败”）。事件日志 `events.jsonl`：`announce_begin / announce_done / wait_begin / wait_end / cancel(reason)`。
> 2. **现场授权独立配置 `config/authz.conf`**（与模型/动作资产、`site.conf` 分离）。新包随附**空文件 = 未授权**：服务 **NOT ARMED**（F1+F3 仍完整只读检查；L1+A 不动作并语音“**未授权**”）。由人在现场显式登记：`./authorize.sh --phrase I-ACCEPT-UNVALIDATED --confirm-protection [--mode dance] [--hours 24]`（等价 `./start.sh --authorize …`；全部是命令行参数，**没有任何交互提问**）。登记行绑定 `policy.onnx` sha256、`motion.bin` sha256、模式、参考模式、最大窗口、最大 authority、主机、有效期（默认 24 h，最长 168 h）、口令摘要和包构建（可执行文件/配置哈希）：换模型/换动作/重编译/改配置/换模式/换机器/过期 都会失效。`./start.sh --revoke [--mode M]` 撤销，`--authz-show` 查看；服务启动时还会**自行复核**该文件（手敲命令行 + `--accept-unvalidated` 也绕不过）。`--sim` 开发回环里 `--yes --accept-unvalidated` 仍可武装（仅供开发测试）。
> 3. **start.sh 无任何交互输入**（不读 stdin，`< /dev/null` 不阻塞）；默认完整舞蹈；`--revoke / --dry-run / --list / --sim` 保留。
> 4. **日志计数修复**：真机第一次运行 `dropped_states:48784` → `LOG_DROPPED` FAIL 是**误报**：pre-roll 环形缓冲在 READY 等待期间按设计滚动覆盖最旧样本，旧代码把每次覆盖都记成 `dropped_states`。V1.6 把“环淘汰”（`state_ring_evicted`，设计内）与“流式阶段真实溢出”（`dropped_states`）分开计数，并修了 writer 每轮上限的“先取后判”丢 1 条、begin_stream 时环已满导致接管后头几个样本丢失；`lowstate_lost_est` 口径改为按 1 kHz/ms tick 解释并给出控制窗口内最大到达间隔。详见 `LOGGING.md` §8。
> 5. **跟踪/倾角统计去遗留语义**：trace 新增 `dance_frame`（= `ssb_frame − lead_in_frames`，前导为负）、`dance_t`、`seg`（0 前导/1 舞蹈/2 末帧保持/3 暂停参考）；`run_report.tracking_segments` 分别给出 前导 / 舞蹈 / 末帧保持 / 接管首 tick 统计；`TILT` 判定改为**参考感知**：报告 实测倾角、参考倾角、重力方向偏差，仅当偏离参考或超绝对硬上限才 FAIL（阈值取自运行时判据 `tilt_judge_config`）。**真机三次日志中“最大倾角 0.42 rad > 0.40 FAIL”是舞蹈本身的前倾（参考倾角 0.40），偏差仅 0.02 rad。**


> **V1.5.1（部署热修复，依据首次真机 CHECKING 阶段崩溃日志 `logs.zip`）**
> 1. **DDS 崩溃根因与修复**：日志 `EntityDelegate::prevent_callbacks(): Assertion 'false' failed`（SIGABRT/134，F1+F3 之后 1.3 s，CHECKING 阶段）。CycloneDDS-C++ 0.10.2 在**销毁一个带 data_available 监听器的 DataReader 时恰好有回调在途**会断言（回调线程持有最后一个引用，自己跑析构）。旧代码每次 F1+F3 / L1+A 都**新建并销毁** 3 个临时 reader（`rt/lowcmd` 厂家服务 ~500 Hz，必然有回调在途）。现在：**进程内任何 DDS 实体都不再被销毁**（常驻监视 reader + 10 s 滚动历史做流量判别，回调只持有共享的 Core，`Link` 析构只“停用回调并把实体寄存到进程生命周期的墓地”）；`dev/dds_stress` 复现了同一断言并验证修复（600 次扫描 + 60 次 Link 创建/销毁循环无崩溃）。
> 2. **预热 runner**：服务启动时（以及每次运行结束回到 IDLE 后）就启动一个 STANDBY runner：硬性拒绝检查、参数/动作加载、DDS 连接与监视、ONNX 加载、推理预热都已完成，**不持有锁、无命令 writer、不发任何 RPC**。F1+F3 只写触发文件，runner 随即做实时检查（最近 1 s 滚动历史、GetFsmId、guardian、冻结哈希**并行全量复算**）→ 准备就绪；L1+A 之后的 GO 复核 / ARM 预检（前向 0.3 s 窗口）/ 接管 / 退出流程**不变**。每次运行仍是全新进程（Policy 历史、上一动作、参考索引重建）。预热 runner 异常退出 → 自动重启（退避，3 次快速失败后停用并回退到冷启动）；F1+F3 时没有预热 runner 则走原冷启动流程。`--no-standby` 可关闭。
> 3. **start.sh**：默认加载**完整舞蹈**（`dance`），没有交互式测试类型菜单（`--mode`/`--list` 保留）。未验证包络采用**一次性现场授权**：第一次键入 `I-ACCEPT-UNVALIDATED`，二进制写入绑定“本包构建（含可执行文件哈希）+ 模式 + 窗口 + 主机”的记录（`auth/`，默认 24 h，最长 168 h）；之后启动不再询问。换包/重新编译/改 site.conf、modes.conf/换模式 → 重新询问；`./start.sh --revoke` 撤销；`--no-remember` 恢复每次询问。site.conf 不能授予授权；无有效授权时仍是 NOT ARMED（F1+F3 做完整只读检查，L1+A 无效）。
> 4. **解压即可执行**：zip 里所有脚本 0755（含文件类型位和目录条目）；脚本启动时给同目录脚本补 +x，丢权限位的解压工具也能 `bash start.sh`。
> 5. **计时**：`plan_timeline()` 单一函数规划 前导冻结 + 整段动作 + 末帧保持，runner、selfcheck 7k 与打包器共用；日志里打印 `timeline:` 一行，manifest.json 记录 `planned_total_s / failsafe_s / covers_whole_dance`。
> 6. 策略、冻结哈希、控制律、安全阈值、88 N·m 限幅、guardian、双击 START、L2+B、WalkRun 优先恢复、SETTLE/HOLD 状态机均**未改**。
>
> **P5.3**（依据 P5.2 完整舞蹈三次实机日志）：① **可恢复退出自动回 IDLE，无需 `ack.sh`**；② 修复 guardian 正常退出被误判 CRITICAL（误杀 runner 的 post-exit 记录）；③ 排查“>95% 指令裁剪”：**主因是 Policy 本身输出越限的踝目标，部署端安全门把目标裁到限位内使踝力矩少于训练**（详见 `P5_3_ANALYSIS.md`，在 `results/p5/`）；④ `running` 模式默认补 1 s 前导冻结（复现训练片段）；⑤ 实验选项 `--torque-ff`（默认关）；⑥ 日志统计分开 Policy 原始输出 / 裁剪后指令 / 实际位置 / 参考，并修正“触发帧未计入最大倾角”。Policy、控制律、安全阈值、急停、WalkRun 优先恢复、遥控器操作与语音均不变（仅新增 1 句“可再次开始”）。
>
> **服务状态机（新）**：正常完成 / 手动退出 / **可恢复**异常退出（WalkRun 已确认，原因不是 ESTOP·L2+B·倒地·状态不可信·FSM 被改）→ `SETTLE`：等 **WalkRun(500/801/802) 确认 + 姿态安静（倾角<0.15 rad、|gyro_xy|<1.5 rad/s 持续 1 s）+ 控制锁释放 + guardian 就绪**，满足后自动回 `IDLE`，语音“可再次开始”，**可直接再次 F1+F3 / L1+A，不需要 `ack.sh`**；20 s 内不满足 → `HOLD`（原因写明缺哪一项）。**不可恢复**（紧急被动、倒地、WalkRun 恢复失败、退出未确认、runner 崩溃由 guardian 接管、通信/状态异常等）仍进入 `HOLD`，必须检查后 `./ack.sh`。**任何情况下都不会自动接管**：回到 IDLE 后仍要重新 F1+F3 → READY → L1+A。每次运行都是新的 runner 进程（Policy 历史、上一动作、参考帧索引全部重建，日志 `policy_state_reset` 记录并校验“clean”）。

> **P5.2 紧急修复**（现场 `logs(3).zip`）：`check` 模式在没有运动授权时，runner 仍套用运动准入门，3 次 F1+F3 全被 `REFUSED`（rc=2），又被误记成 `CRASH_NO_REPORT`。修复：① `check` 是纯只读模式（runner 以 `--check-only` 启动，永不接管、不加运动授权门）；② 删除模式间顺序依赖，历史 PASS/FAIL 仅供参考；③ 授权只问一次（hold=YES，未验证=I-ACCEPT-UNVALIDATED）；④ 拒绝写 `REFUSED` 报告和真实原因；⑤ 回归覆盖所有模式的真实 `start.sh` 入口。Policy、控制律、安全阈值、急停、WalkRun 恢复、日志均未改。

**冻结**：Policy = `robust_d it12000`（sha256 `06f02449…3b01`），参考动作 `motion.bin`、`policy_params.txt`、控制律、安全阈值、WalkRun 退出策略均不变；本阶段不训练、不搜参。
本阶段新增：① 统一的**测试模式表**（`config/modes.conf`）和一步启动；② **完整日志系统**（见 `LOGGING.md`）；③ `status.sh` / `export_logs.sh`。
**本包不会自动连接或控制真机**：`./start.sh` 只监听遥控器，机器人要等你按键。

当前真机证据（用户提供）：稳定接管 3/3 PASS、WalkRun 正常恢复 3/3 PASS；**主动退出与动态舞蹈尚未验证**；足底穿透（原生 MuJoCo 17.4 mm，门槛 5 mm）仍超标——舞蹈类模式必须吊带保护。

## 1. 安装（机载电脑 aarch64，约 1–3 分钟，不连接机器人）
```bash
unzip partymimic_cpp_p5.zip && cd partymimic_cpp
./build.sh        # 编译 + 离线自检（selfcheck 必须全 PASS，否则不要上机）
```

## 2. 现场操作（机器人先用原厂遥控器站稳在 WalkRun，FSM 802）
```bash
./start.sh                      # V1.5.1：默认完整舞蹈（dance），无菜单；其它模式：./start.sh --mode hold|exit|clip2|clip4|check ；./start.sh --list
```
**各模式互相独立：任何模式、任何顺序都可以直接启动，没有前置条件，不需要 `--skip-order`。** `logs/mode_pass.jsonl` 的历史 PASS/FAIL 只用于记录和菜单提示，**绝不作为启动条件**。

**授权（V1.6：没有任何提问，全部是命令行参数 / 独立登记文件）：**
| 模式 | 授权 |
|---|---|
| `check`（只读检查） | **无需任何授权**，永不武装；F1+F3 只做检查，**绝不下发电机指令**，L1+A 无效 |
| `hold`（已验证包络：首帧冻结、≤3 s） | `./start.sh --mode hold --yes`（吊带/急停在手/≥2 人） |
| `exit` `clip2` `clip4` `dance`（真机未验证） | 现场登记一次：`./authorize.sh --phrase I-ACCEPT-UNVALIDATED --confirm-protection`（写入 `config/authz.conf`，绑定本模型/动作哈希）；之后 `./start.sh` 自动武装 |
| 以上任一没给 / 登记失效 | **NOT ARMED**：服务照常启动，F1+F3 仍完整执行只读检查，L1+A 无效（语音“未授权”） |
`--dry-run` 只打印将要启动的参数（含授权结论）；`./start.sh --revoke` 撤销登记。

| 操作 | 作用 | 语音 |
|---|---|---|
| 启动 | 服务监听 | “服务已启动” + 模式名（仅检查/保持测试/退出测试/舞蹈二秒/舞蹈四秒/完整舞蹈） |
| **F1+F3** | 加载 Policy + 实时准入检查（机器人不动） | “正在加载” → “准备就绪” 或 “检查失败” |
| **L1+A** | 仅已武装时：松开所有键 → 播报“开始执行” → **播报完成 + 0.5 s（announce_wait_s）→ 重新确认前提 → 接管** → 执行 → 自动交还 WalkRun；等待期间的任何取消/急停/故障都不接管 | “开始执行” → “执行完成”；未授权时“未授权” |
| **双击 START** | 任何阶段中断；接管前什么都不发，接管后**立即**请求 WalkRun 并确认 | “已退出” |
| **L2+B**（≥0.1 s） | 仅运行期：紧急 PASSIVE（独立通道，不经 WalkRun） | “紧急被动” |
| 故障 / 失稳 | 自动立即请求 WalkRun，FSM 确认后报告，服务进入 HOLD | “异常已交还” / “恢复失败” |
| `./stop.sh [--estop]` / `./status.sh` / `./ack.sh` | 停服务（运行中立即交还 WalkRun）/ 状态 / **仅 HOLD** 后确认 | |

舞蹈类模式（`clip2/clip4/dance`）默认在舞蹈前冻结首帧 1 s（`site.conf lead_in_s`，复现训练片段；`run_s` 仍是舞蹈秒数）。实验选项 `./start.sh --mode clip2 --torque-ff`：目标仍裁在限位内，把训练等效力矩作为前馈发出（默认关，**未在真机验证**，需要未验证授权）。

## 3. 测试模式（`./start.sh --list`；建议顺序仅供参考，不强制）
| 模式 | 内容 | 期望结果 |
|---|---|---|
| `check` | 只读 F1+F3 检查 | READY，结果 `CHECK_PASSED`，admission 无 FAIL |
| `hold` | 接管后保持 SSB 第 0 帧 3 s（真机 3/3） | `COMPLETED`，WalkRun 确认 |
| `exit` | 保持第 0 帧最长 8 s，**运动中双击 START** 验证主动退出 | `EXITED_WALKRUN`（REMOTE_STOP），WalkRun 确认 |
| `clip2` / `clip4` / `dance` | 舞蹈前 2 s / 前 4 s / 整段 7.1 s（+0.5 s 冻结） | `COMPLETED`，WalkRun 确认 |
拒绝启动也是一条**结果**：runner 因参数/授权拒绝时写 `run_report.json`（`result=REFUSED`，`exit_reason` 为真实原因），不会再被记成 `CRASH_NO_REPORT`。

## 4. 准入条件（F1+F3 与 L1+A 复核；FAIL 阻断，WARN 只记日志）
**FAIL（阻断）**：lowstate 中断（无数据/<15 Hz/tick 不走）· 状态越过限位或护栏 · 入场倾角 ≥0.40 rad · 入场 FSM 不在 {500,505,801,802}（0/1 恒拒绝）或 GetFsmId 失败 · mode_machine 不符 · **有源竞争控制器**（`rt/user_lowcmd`/`rt/arm_sdk` 实际有流量）· 控制权锁/急停锁占用 · guardian 不在线 · policy/params/motion 的 sha256 不等于冻结值 · L1+A 时 FSM 与 READY 时不同 · **日志目录剩余空间 <100 MB 或日志无法打开（新增：记录不了的运行不开始）**。
**WARN**：lowstate 15–30 Hz、机器人未完全静止、推理 >15 ms、`rt/lowcmd` 无基线/无流量。
本程序**不会**替你切 FSM：先用原厂遥控进入 WalkRun。

## 5. 停止条件
**立刻停止整个测试、`./export_logs.sh`、回到离线分析（不要再试下一步）：**
- 任一次结果不是该模式的预期结果：`WALKRUN_RECOVERY_FAILED`、`FALLEN_PASSIVE`、`EMERGENCY_PASSIVE`、`HANDBACK_UNCONFIRMED`、`ARM_FAILED`、`CRASH_NO_REPORT`，或服务进入 HOLD；
- `anomaly_summary.txt` 为 FAIL（含 `LOG_DROPPED`/`PROCESS_CRASH`），或 WARN 中出现倾角 >0.25 rad、关节速度/力矩触顶、`CMD_NOT_FOLLOWED`（先看日志再决定是否重做同一模式）；
- 现场异常：异响、抖动、过热、支撑带受力异常、非预期姿态 → 立即 **L2+B**，再 `./stop.sh --estop`；
- 机器人未回到 WalkRun（语音“恢复失败”/“检查失败”）→ 不要靠近，用遥控器阻尼/断电，`./ack.sh` 之前先看 `logs/latest/`。

系统自动退出（无需人工）：推理/NaN/超时/看门狗/关节超速/力矩持续超载/通信中断/失稳（倾角>0.40，或>0.30 且角速度>8 rad/s 或倾角变化率>3 rad/s）→ 立即请求 WalkRun；倾角>1.0 rad（已倒地）→ PASSIVE。退出策略详见 `EXIT_POLICY.md`，安全门审计见 `SAFETY_AUDIT.md`。

## 6. 日志与导出
每次运行自动生成 `logs/runs/<时间>/`（`logs/latest` 指向最近一次）：`run_report.json`、`anomaly_summary.txt`（异常摘要）、`trace.csv`（逐 tick 完整记录）、`obs.bin`、`state_stream.bin`（接管前 5 s + 退出后 2 s 全速率 lowstate）、`events.jsonl`、`safety_clips.csv`、`exit_frames.csv`、`manifest.json`（程序/配置/policy/动作文件哈希）。格式与用法见 `LOGGING.md`。
```bash
./export_logs.sh                  # 一键打包全部日志 -> export/partymimic_logs_<主机>_<时间>.tar.gz (+ .sha256)；崩溃的运行会先自动补写报告
./export_logs.sh /media/usb       # 指定输出目录；--light 不含 obs.bin/state_stream.bin（只能快速浏览，不能回放）
build/partymimic logtool check logs/latest      # 完整性检查（机载，不连接机器人）
build/partymimic logtool replay logs/latest     # 逐 tick 回放 Policy（同机逐位一致；跨 aarch64/x86 ≤1e-5）
```
开发机分析（`partymimic_p5_analysis.zip`，numpy + onnxruntime）：`python3 analysis/pm_analyze.py <解压后的 run_dir> all`。

## 7. 常用参数与配置
`config/site.conf`：入场 FSM、语音、`max_run_s`/`max_authority`、退出与安全门阈值、`log_*`。**`max_run_s` 默认 12 s，不得超过编译期上限 `ABS_MAX_RUN_S`（60 s）；动作窗口 `run_s` 必须 ≤ `max_run_s` ≤ 60。比 12 s 更长的舞蹈：在 `site.conf` 显式写 `max_run_s <整段秒数>`（`./pm package` 会自动这样做并记入 `ADMISSION.json`），真机仍需 `--accept-unvalidated`；`lead_in_s` 的 1 s 冻结首帧在 `run_s` 之外。**`config/modes.conf`：测试模式表（唯一定义处）。
`./start.sh --help`；`--iface IFACE --entry-fsm N --foreground --no-tts --yes --dry-run --sim --authorize/--revoke/--authz-show`（`--accept-unvalidated` 只在 `--sim` 开发回环里有效）（`--sim` 仅开发机 loopback 机器人）。
不可恢复故障后服务进入 HOLD：检查机器人和日志 → `./ack.sh`；可恢复退出会自动回到 IDLE（见上）。

## 8. 包内容（严格白名单，无 Python / 模型 XML）
```
build.sh start.sh authorize.sh stop.sh ack.sh status.sh export_logs.sh CMakeLists.txt
README.md LOGGING.md EXIT_POLICY.md SAFETY_AUDIT.md
config/site.conf config/modes.conf config/authz.conf（空：未授权）
src/*.cpp include/pm/*.hpp include/gamepad.hpp
assets/policy.onnx policy_params.txt motion.bin fk_chain.txt SHA256SUMS
third_party/onnxruntime/        ONNX Runtime 1.22.0 aarch64
```
单一可执行文件 `build/partymimic`：`service` / `runner` / `guardian` / `release` / `selfcheck` / `logtool`。
