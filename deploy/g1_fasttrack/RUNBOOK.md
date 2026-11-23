# PartyMimic G1 FastTrack — 运行手册（实机前置版）

状态：**Sim2Sim 通过；实机 NOT AUTHORIZED（尚未接触机器人）**。本手册只描述“只读预检 → 悬吊短片段 → 异常退出”。
禁止：落地整段舞蹈、使用 P3.x 候选轨迹、跳过任何一步。

## 0. 一句话架构

`EGalahad/sim2real` 的 `Tracking` 推理栈（ONNX + `supermimic.*` 观测 + 冻结 SSB）原样复用；只替换传输层与控制流：

```
rt/lowstate ──► ControlSession ──► SafetyGuard ──► rt/user_lowcmd      (官方 SDK2 用户控制通道)
 'sport' RPC:  GetFsmId(7001) / SwitchToUserCtrl(7110) / SwitchToInternalCtrl(7111)   (官方 LocoClient)
```

**不使用** EGalahad 的 `MotionSwitcher.ReleaseMode + rt/lowcmd` 调试模式路线（那条路线无法把控制权还给原厂控制器）。
SDK 固定为官方 `unitree_sdk2_python@814556d`（EGalahad 的 fork 没有 `g1/loco`），隔离安装在 `.sdk_target/`，不污染环境。

## 1. 最简命令

```bash
cd deploy/g1_fasttrack
bin/ft test                 # 23 个单元测试
bin/ft sim-smoke            # 仿真冒烟（回环网卡 lo，不碰任何真实网络）
bin/ft sim-dance            # 悬吊仿真，整段，live
bin/ft sim-faults           # 27 项安全门槛矩阵
```

## 2. 切换方法（OEM → 用户程序 → OEM）

| 阶段 | 动作 | 依据 |
|---|---|---|
| 入口条件 | `GetFsmId == 1`（PASSIVE，官方示例唯一演示的入口） | 官方 `g1_userctrl_dds_example.cpp` |
| 进入 | 先发 1 帧阻尼（kp=0,kd=1.5）→ `SwitchToUserCtrl()` → 轮询 `GetFsmId`，**fsm 变化才算确认** | 同上；ACK 语义官方未公开，见 §6 |
| 之后 | 阻尼持续发送直到确认；然后 PREPARE（增益与位置平滑过渡到 SSB 首帧姿态） | — |
| 退出 | 阻尼 → `SwitchToInternalCtrl(LAST)` → 轮询 `GetFsmId` 确认 | 官方示例用 `LAST` |
| 退出失败 | 显式拒绝（3203/3204/3102）才重试 **1 次**；超时（3104）**不重试**；仍未确认 → 持续阻尼并大声报 `EXIT_UNCONFIRMED`，退出码 5 | 超时≠未执行 |

`--exit-mode PASSIVE|WALKRUN`、`--allowed-entry-fsm`、`--expect-user-fsm` 可改，但**必须先由 OEM 书面确认**。

## 3. 实机三步（每步都是新的授权点）

所有真机命令都需要：`PARTYMIMIC_G1_ALLOW_MOTION=HANGING_SHORT_SEGMENT_ONLY`、`--i-confirm-hanging-and-estop`、
15 分钟内的只读预检通过文件、**正在运行的 guardian**、`segment ≤ 3 s`、`authority ≤ 0.3`。缺任何一项 runner 直接退出码 2，且不创建任何 DDS 写者。

### 步骤 A：只读预检（不运动，不创建 publisher，只调用 GetFsmId）
```bash
bin/ft preflight <机器人网卡>        # 通过后写 runs/preflight_pass.json
```
检查：状态流速率/tick 前进、状态在护栏包络内、关节在限位内、静止、倾角、FSM=允许入口、用户控制话题上**无其他写者**、独占锁空闲、急停文件不存在、ONNX/SSB SHA 冻结。
同时**人工核对关节顺序**：手动轻动一个已知关节（例如左肘），确认 `rt/lowstate` 里 index 18 变化（脚本会打印 q）。

### 步骤 B：悬吊 shadow（进入用户控制，移动到 SSB 首帧姿态并保持，策略只计算不执行）
终端 1：`bin/ft guardian <网卡>`　终端 2：`bin/ft hang-test <网卡>`
通过标准：`SwitchToUserCtrl` 返回 0 且 fsm 变化；SETTLE 通过（指令有效）；shadow 偏差 < 1.2 rad；退出确认；全程无故障。

### 步骤 C：悬吊 live（≤3 s，权限 ≤0.3）
终端 1：guardian　终端 2：`bin/ft hang-live <网卡>`，到 WAIT_GO 时由操作员按 `g`。
任何时刻：`x` / 空格 / ESC / Ctrl-C = 急停；另一终端 `bin/ft estop <网卡>`（直接接管）；遥控器急停始终在手。

## 4. 异常与退出流程（已在 Sim2Sim 验证，见报告）

| 情形 | 系统行为 | 操作员 |
|---|---|---|
| 护栏故障 / 急停 / 看门狗 / 通信超时 | 立即 100 Hz 阻尼，锁存故障，`SwitchToInternalCtrl`，确认 | 等 `released confirmed=True` |
| 进程被 kill -9 / 卡死(SIGSTOP) / 崩溃 | **guardian** 检测（0.3 s 无指令或进程消失而 FSM 仍在用户态）→ kill 残余进程 → 阻尼 → 退出 RPC → 确认 | 无需操作，观察 guardian 输出 |
| 退出 RPC 无法确认（`EXIT_UNCONFIRMED`） | 持续阻尼，退出码 5 | **遥控器阻尼/急停，吊绳托住，不得靠近，必要时断电** |
| 机器人端无响应（host_dies 类） | 同上 | 同上 |
| 手动急停锁 | `touch /tmp/partymimic_g1_ESTOP` 或 `bin/ft estop`；清除用 `bin/ft estop-clear`（确认机器人安全后） | — |

## 5. 实机前必须由人完成的事项（软件无法替代）

1. 机械：悬吊点承重、双脚离地 ≥10 cm、吊带不与手臂/腿干涉，旁人距离 ≥2 m，遥控器急停在手，2 人在场。
2. 电气：确认 G1 型号、DOF/腰锁定配置、运控固件版本；**把版本号填入报告**（目前 UNKNOWN）。
3. OEM 书面确认：`SwitchToUserCtrl` 的 FSM 值、`UserCtrl` 下 `mode/mode_machine/CRC` 是否需要、用户指令的超时/看门狗行为、`LAST` 的恢复语义。
4. 若 OEM 要求 `rt/lowcmd` 风格字段，使用 `--lowcmd-fields lowlevel`（置 `motor.mode=1`、`mode_machine`、CRC）。默认 `example` 严格对应官方用户控制示例（只设 q/dq/tau/kp/kd）。
5. 用 `bin/ft preflight` 在**真机**上通过（本次只在回环仿真上自测）。

## 6. 已知未知项（诚实清单）

- 官方 `SwitchToUserCtrl` 的完成语义/FSM 数值未公开；本实现用“fsm 变化”作弱确认，并在 `--expect-user-fsm` 可设强确认。
- `rt/user_lowcmd` 的 watchdog/lifespan 未公开；仿真宿主默认**假设无 watchdog**（最坏情形），不依赖它。
- G1/G1+ 兼容性、固件下限：未知。
- 力矩护栏用 PD 估计力矩（`kp·e − kd·dq`）限幅；电机热/电流/速度包络未建模。
- P2 动力学安全 FAIL（穿透 12.5 mm > 5 mm）与 P3.9 NO-GO 仍然成立：**舞蹈接触可行性未被证明，故只允许悬吊**。
