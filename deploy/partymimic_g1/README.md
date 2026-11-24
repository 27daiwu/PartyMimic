# PartyMimic G1 真机部署包（精简版）

```bash
unzip partymimic_g1.zip && cd partymimic_g1
./build.sh      # 在机载电脑(aarch64)上离线编译安装，约几分钟；不连接机器人
./start.sh      # 启动遥控器监听服务（会让你确认防护；不会自动动机器人）
```

## 遥控器（机器人先用原厂遥控器站在 WalkRun）
| 操作 | 作用 |
|---|---|
| **F1+F3** | 加载 Policy + 实时安全检查 → `READY` 或 `BLOCKED`（此阶段机器人不动） |
| **L1+A** | （READY 后）从 WalkRun 接管 → 执行 SSB 动作 → 自动恢复 WalkRun |
| **双击 START** | 任何阶段安全退出（接管前：什么都不发；接管后：阻尼并归还 WalkRun/PASSIVE） |

键位沿用 `wbc_nx/remote_start` 已验证的 `wireless_remote` 布局（F3 = SDK 的 F2 位）。按键无反应时看 `logs/keys.log`（每次按键变化的原始字节）。
`./start.sh` 提示时输入 `YES` 才会“武装”（防跌倒保护到位、遥控器急停在手、≥2 人）；否则只做检查，L1+A 无效。
可选参数：`--iface eth0 --entry-fsm N --run-s 1.0 --ref running|paused --authority 0.3 --foreground`。真机上限：动作窗口 ≤1.0 s、权限 ≤0.3（整段舞蹈仍被禁止）。

## 安全机制（始终在线，代码强制）
控制权互斥（独占锁 + `rt/user_lowcmd` 外部写者检测 + 急停锁）· 关节硬限位（入场零容差 + 运行中护栏）· 通信超时/卡死（lowstate>60 ms、控制环看门狗 100 ms）· 失衡保护（倾角>1.0 rad、角速度、关节速度/力矩）· 独立 guardian 进程（runner 崩溃/挂起/START 未响应时接管）· 安全退出（阻尼 → `SwitchToInternalCtrl` → `GetFsmId` 确认；姿态异常→PASSIVE）。

## 日志（全部在 `logs/`）
`service.log` 总日志 · `keys.log` 按键 · `events.jsonl` 状态/控制权事件 · `guardian/` · `runs/<时间>/`：
`trace.csv`（50 Hz：关节 q/dq/力矩、IMU、Policy action/q_policy、q_cmd、推理耗时、SSB 帧、遥控字节）、`events.jsonl`（锁、SwitchToUserCtrl、确认、急停、SwitchToInternalCtrl、释放）、`ready.json`（READY 时状态/fsm/DDS 写者/各检查项）、`run_report.json`（结果、**退出原因**、故障、归还确认、进入/恢复 fsm）。`logs/latest` 指向最近一次。
故障后服务进入 HOLD：检查机器人和日志 → `./ack.sh`。`./stop.sh` 停服务（`--estop` 同时置软件急停）。

## 剩余真机运动阻塞（状态：BLOCKED）及最小解决方案
| # | 阻塞 | 最小解决方案 |
|---|---|---|
| 1 | **`rt/lowcmd` 写者身份**（R1 诊断项，代码保留为实时检查）：WalkRun 下 OEM 运控服务本来就写它，软件无法判定，`config/site.json` 默认未确认 → `READY` 前即 `BLOCKED` | 机器人站在 WalkRun 按 F1+F3，把 `logs/runs/*/ready.json` 的 `writers` 发给 OEM，书面确认后在 `config/site.json` 填 `confirmed:true, expected_writers:<数量>`；之后写者数量变化仍会拦截 |
| 2 | **入场姿态**：此前两次只读诊断时机器人在 fsm 0/1 的折叠姿态（膝 2.907 > 限位 2.880 rad）。检查保留（fsm 必须是 500/505/801，q 在硬限位内） | 用遥控器站到 WalkRun 再 F1+F3；若站立时膝仍超限，把 `ready.json` 发给 OEM 核实行程，**不要改限位** |
| 3 | **固件行为未实测**：`SwitchToUserCtrl` 后 fsm 是否 1000（否则 `SWITCH_FAILED`→阻尼归还）、退出后是否回到进入时的 fsm（报告 `restored_to_entry_fsm`）、**用户控制期间 `wireless_remote` 是否还更新**（决定运动中双击 START 是否有效；看 `trace.csv` 的 `keys_b2/b3`），用户指令看门狗 | 首次只在吊带/龙门架保护下用 `--ref paused --run-s 0.5` 跑一次，按日志核对上述项；若 START 在运动中无效，运动期间只靠遥控器急停 + guardian |
| 4 | **动作范围**：P2 动力学安全 FAIL（穿透 12.5 mm > 5 mm），仿真 frame0 入口+权限 0.3 时倾角可达 0.7 rad（护栏 1.0 rad）；WalkRun 对被扰动机器人的接管语义未知 | 保持 ≤1.0 s / ≤0.3；放宽需另行评审，不在本包内 |

仿真证据（服务端到端 26/26、单元测试 133）保留在开发工程 `deploy/g1_fasttrack`，现场不再运行。
