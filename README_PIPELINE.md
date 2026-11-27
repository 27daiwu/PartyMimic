# PartyMimic Pipeline V1.1 使用说明（含 V1.5 / V1.5.1 增补，见 §15、§16）

一条命令完成：`robot_motion.pkl` → 数据转换/QA → robust_d 训练 → 自动评估筛选 → ONNX → Sim2Sim → C++ Sim2Real 部署包 → 日志分析。
**全程离线，不会连接真机。** 所有产物按舞蹈名隔离在 `results/<舞蹈名>/`，SSB 历史成果（`results/p0…p5`、`deploy/partymimic_cpp`、`src/`、`tools/`）只读复用，不被修改。

```bash
cd /home/unitree/workwpace/zjy_ws/PartyMimic
./pm doctor                      # 环境自检（venv / GPU / tmux / C++ 框架 / 数据集）
./pm --help
```

## 1. 一键全流程
```bash
./pm run <舞蹈名> --iters 40000          # 例：./pm run ythn --iters 40000
tmux attach -t pm-<舞蹈名>               # 看进度（窗口：train / eval / tensorboard / monitor / pm-pipeline）
./pm status <舞蹈名>
```
`run` = prepare → train（含 2000 轮短程闸门）→ 评估 → 平衡优先筛选 → export → sim2sim → package → analyze。
Sim2Sim 不通过则**不打包**、准入记为 `FAILED`。已有同配方训练在跑/已完成时会**接管**而不是重复训练（如 `ythn`）。
冒烟（100 轮、极小评估预算，只验流程）：`./pm run <舞蹈名> --smoke`

## 2. 分步命令
```bash
./pm prepare <舞蹈名>                 # 读 /home/unitree/motion_datasets/video/<舞蹈名>/robot_motion.pkl → 50Hz npz/csv + QA + padded 训练片段
./pm prepare <舞蹈名> --source /path/other.pkl --force   # 换源文件（如 robot_motion_twist.pkl）
./pm preview <舞蹈名> [--video]       # QA 摘要 + 预览图(results/<舞蹈>/reports/*.png)
./pm train   <舞蹈名> --iters 40000   # tmux 训练 + 自动续训监督器 + 评估守护 + TensorBoard
./pm status  <舞蹈名> [--watch]
./pm stop    <舞蹈名>                 # 优雅停止，保留 last.pt
./pm resume  <舞蹈名> [--iters 60000] # 断点续训；加大 --iters 即延长训练
./pm eval    <舞蹈名> [--select]      # 补评 checkpoint；--select = 硬复评 + 平衡优先筛选
./pm eval    <舞蹈名> --ckpt 20000 30000 [--hard]    # 评估指定权重
./pm export  <舞蹈名> [--it 32000]    # ONNX + 一致性校验 + 部署资产（默认导出筛选出的最终模型）
./pm sim2sim <舞蹈名>                 # 筛选→导出→Sim2Sim→写准入记录；失败返回码 1
./pm package <舞蹈名>                 # 生成 C++ 部署包 zip（准入未 PASS 则拒绝）
./pm analyze <舞蹈名>                 # 训练/评估分析 → analysis/TRAINING_ANALYSIS.md
./pm analyze <舞蹈名> --logs <真机 export_logs 的 tar.gz 或 logs/runs/<时间> 目录>   # 真机日志分析
./pm list                             # 已有舞蹈 / run
```
同一舞蹈可有多个 run：加 `--run-id <名>`（默认 = 训练配置名）。

## 3. 训练要点
- **tmux**：会话 `pm-<舞蹈名>`；`train` 窗口运行监督器（崩溃/卡死/NaN 自动恢复，异常回滚到备份，最多重启 30 次，磁盘 <15 GiB 中止）。
- **断点续训**：每 1000 轮写 `last.pt`（完整状态）。断电/关机后 `./pm resume <舞蹈名>`。
- **GPU 自动检测**：选空闲显存最多的卡；显存不足自动把 `num_envs` 降档（4096→2048→…）并告警；GPU 已被别的训练占用会拒绝，除非 `--allow-concurrent`。手动：`--gpu 0 --num-envs 2048`。
- **轮次**：`--iters N`（默认取训练配置的 40000）。
- 起点 = 官方预训练 `checkpoints/g1_mlp.pt` 经 `tools/p4_ckpt_surgery.py` 加宽，再在该舞蹈上微调（与 SSB/ythn 已验证做法一致）。

## 4. 配置（版本化，训练/部署/评估三者解耦）
| 目录 | 内容 | 默认 |
|---|---|---|
| `pipeline/configs/train/` | 怎么训练（robot 变体、域随机化、奖励、优化器、监督器参数） | `robust_d_v1` |
| `pipeline/configs/deploy/` | 怎么上机（C++ 框架、测试模式时长、site.conf 覆盖、准入规则） | `cpp_g1_v1` |
| `pipeline/configs/eval/` | 怎么评估/筛选/Sim2Sim 门槛 | `balance_first_v1` |

优化时**复制成新文件**（如 `robust_d_v2.yaml`），用 `--train-cfg robust_d_v2 --run-id v2` 训练；不要原地修改。每个 run 在 `results/<舞蹈>/…/pm/config_snapshot/` 冻结一份副本和 sha256（`manifest.json`），之后改 YAML 不影响已有 run。部署配置可单独换：`./pm package <舞蹈> --deploy-cfg <名>`，无需重训。

## 5. 自动评估与筛选（动态平衡优先，其次还原度）
1. 训练中评估守护按 `eval.periodic`（默认每 2000 轮，固定工况网格：延迟/KP/摩擦/力矩上限/质量负载/推力/初态扰动）逐个评估，写 `eval/summary.csv`、`best/best_{balance,fidelity,combined}.pt`、TensorBoard。
2. 训练结束后对候选做**硬复评**（不同种子、24 次重复、超出训练分布的苛刻工况）。
3. 规则：nominal 100% 存活且鲁棒存活 ≥ 0.90 才合格；合格者中平衡分在最优 2 分以内视为同档，同档取还原度最高者。
4. 主选未过 Sim2Sim 时，按同一规则顺延下一个候选（最多 3 个）。
- 报告：`results/<舞蹈>/…/reports/SELECTION_REPORT.md`；最终选择：`final/final_selection.json`。

## 6. Sim2Sim 与真机准入
- Sim2Sim = C++ guard 等价控制链 + 200 Hz PD + MuJoCo，初态取自真机 WalkRun 站姿（`pipeline/ref_data/walkrun_real_runs`），15 种工况（延迟/摩擦/KP/推力/噪声）；**每工况 24 个初态 = 10 个真机站姿 + 14 个插值采样**（详见 V1.2「Sim2Sim 口径」）。
- 准入记录 `pm/admission.json`：`real_robot_admission` 仅当 评估合格 ∧ ONNX 一致性 ∧ Sim2Sim 门槛 全部通过才为 `PASS`；否则为 `FAILED / NOT_ELIGIBLE`，`./pm package` 直接拒绝（返回码 5）。
- **`PASS` 只表示“离线门槛通过，允许进入有保护的真机试验”，不是真机结论**——部署框架的授权口令（`YES` / `I-ACCEPT-UNVALIDATED`）仍然必须现场输入。
- 仅供离线检查可用 `--allow-unadmitted`：包名带 `_UNADMITTED`，内含 `UNADMITTED_DO_NOT_DEPLOY.txt`。

## 7. 部署包（C++ Sim2Real）
```bash
./pm package <舞蹈名>      # → results/<舞蹈>/…/deploy/partymimic_cpp_<舞蹈>_<run>_it<N>_cpp_g1_v1.zip (+ .sha256)
```
- 直接复用最新 `deploy/partymimic_cpp`（遥控器 F1+F3 / L1+A / 双击 START / L2+B、可恢复退出自动回 IDLE、guardian、日志系统）；文件白名单/黑名单读取自框架自己的 `dev/make_zip.sh`。
- 只替换资产：`policy.onnx / policy_params.txt / motion.bin / fk_chain.txt`、编译进二进制的 sha256 常量、selfcheck 黄金动作、`modes.conf`（按舞蹈时长生成）。唯一的源码改动：暂存副本里 selfcheck 7f 对 SSB 模式时长的期望改成本舞蹈的模式表。
- 打包时在开发机（x86）编译暂存副本并跑 `partymimic selfcheck`，必须全 PASS；包内 `ADMISSION.json` 记录模型/哈希/Sim2Sim/自检结果。
- 机载（aarch64）使用：`unzip … && cd partymimic_cpp && ./build.sh && ./start.sh --list`，详见包内 `README.md` / `EXIT_POLICY.md` / `LOGGING.md`。
- 动作时长与运行窗口：`dance` 模式窗口 = 整段动作秒数（向上取 0.1 s），**不再截到 12 s**；runner 另在前面补 `lead_in_s`=1 s 冻结首帧，不计入 `run_s`（例如 YTHN：437 帧 = 8.74 s，窗口 8.8 s，完整播放到最后一帧）。
  三层上限必须满足 `run_s ≤ site.conf max_run_s ≤ 编译期 ABS_MAX_RUN_S`：框架默认 `max_run_s=12`（保守，不变）、编译期上限 60 s。动作整段 >12 s 时，`./pm package` 在**该包** `site.conf` 写入显式 `max_run_s <窗口秒数>`，并把理由记入包内 `ADMISSION.json.run_caps` 和 `PACKAGE_INFO.md`（selfcheck 7d 的期望值同步改写）；动作 >60 s 拒绝打包（不会被悄悄截断，需先评审并抬高 `ABS_MAX_RUN_S`）。
  这只是“允许放行”，不是“已验证”：长舞蹈仍属未验证包络，真机每次必须 `--accept-unvalidated`（start.sh 要求键入 `I-ACCEPT-UNVALIDATED`），`site.conf` 无法授予。
  （`prepare` 的 QA 里“>12 s 会被截断”的告警已过时：该告警只是提示，不再代表截断。）

## 8. 日志分析
真机运行后在机载执行 `./export_logs.sh`，把 tar.gz 拷回：
```bash
./pm analyze <舞蹈名> --logs ~/partymimic_logs_xxx.tar.gz
```
对每个 run 调用框架自带 `pm_analyze.py all`（逐 tick 回放一致性、跟踪误差、PD/力矩、时延、安全裁剪、退出时间线），输出在 `analysis/robot_logs/`。

## 9. 目录结构
```
pm                              入口        pm-test   单元测试(14 项)
pipeline/pmpipe/                流水线代码（prepare / supervisor / evald / evaluator / post / package / analyze / cli）
pipeline/configs/{train,deploy,eval}/   版本化配置
results/<舞蹈>/
  motions/<舞蹈>/ …_padded/     50Hz npz/csv；前后各 1 s 静止的训练片段
  metrics/ qa_kinematic.json    数据 QA；源文件 sha256
  runs/<run>/                   train/ eval/ best/ final/ deploy/ analysis/ reports/ logs/ tb/ state/ pm/
                                （ythn 的历史 run 为平铺布局，原地接管，不迁移）
```

## 10. 常见问题
| 现象 | 处理 |
|---|---|
| `source not found` | 检查 `/home/unitree/motion_datasets/video/<舞蹈>/robot_motion.pkl`；或 `--source` |
| 首帧毛刺 | `prepare` 自动检测并丢弃首部跟踪器初始化异常帧（如 ythn 第 0 帧 53 rad/s），源文件不改 |
| 显存不足 | 自动降 `num_envs`；或先 `./pm stop` 别的训练 |
| 训练被打断 | `./pm resume <舞蹈>`（从 `last.pt`） |
| 短程闸门 FAIL | 自动停训，看 `reports/SHORT_RUN_GATE.md`，找到原因再重来，不靠加轮次 |
| 准入 FAILED | 看 `SELECTION_REPORT.md` 的 Sim2Sim 分工况成功率；调整训练配置新开 run |

## 11. V1.1：动作质量分析 / 闭环回放 / 模型对比（`./pm quality`，`./pm run` 末尾自动执行）
```bash
./pm quality <舞蹈名> [--its 4000 32000 40000] [--n-perturb 8] [--no-video]   # 全部产物 → results/<舞蹈>/…/reports/quality/，总报告 reports/QUALITY_REPORT.md
./pm replay  <舞蹈名> [--its ...]                                              # 只重做四宫格回放视频
```
- **回放**：参考（运动学）+ 最多 3 个模型的 MuJoCo 闭环物理（部署控制链），逐帧叠加接触状态（MuJoCo 接触力占体重 %）、倾角、CoM 投影/CoP/接触点、力矩饱和、护栏改写、根 XY 偏差、俯视小地图、帧号。
- **全局 MPJPE 分解**：全局 / XY 对齐 / XY+yaw 锚定 + 根偏差拆为 {参考自身根不一致, 姿态隐含差, 策略滑移}（按“触地脚不动”航位推算）。判别坐标系误差、参考脚滑、策略漂移。
- **参考质量**：静立段标定地面（悬浮量/漂移）、接触可靠度分层（确定触地 <10 mm、确定摆动 >40 mm、其余“不确定”），**动态 ZMP / 支撑域余量只在可靠帧上计算**，并用闭环 CoP 校验运动学 ZMP；难度对比 LAFAN（缓存于 `pipeline/ref_data/lafan_difficulty_v1.json`）。
- **统一选模规则**（`pmpipe/selection.py`，评估守护、硬复评、报告共用）：合格 → 平衡分在最优 2 分内同档 → 同档取还原度最高；`best/best_selected` 是规则结果，`best_balance/fidelity/combined` 仅为分项最大值（诊断）。
- **溯源**：`quality/provenance.json`（起点/动作/ONNX/checkpoint 哈希、训练启动与续训点、src 未提交 diff、评估器版本）。
- 评估器 V1.1 修正：力矩上限按“执行器→关节”映射取值（V1.0 把执行器序号当关节序号，髋Roll 用了 25 N·m 的肩关节上限）。V1.0 结果归档为 `*_v10_buggy`，不可用于力矩/平衡分比较。

## 12. V1.2：限幅对比 / 修复 / 硬件审计 / 溯源快照 / Sim2Sim 口径

### 溯源快照（`./pm snapshot <舞蹈> [--run-id] [--verify]`）
训练**开始/续训时自动执行**（`./pm train|resume|run`；失败只告警，不阻塞训练）。把“复现这次训练所需的一切”存到 `results/<舞蹈>/…/repro/`：
- `train.patch` / `deploy.patch`：相对 git HEAD 的**完整 tracked diff**（`git diff HEAD --binary --full-index`，二进制安全；训练/流水线树与 C++ 部署树分开）。
- `untracked/` + `untracked_manifest.sha256`：所有未被 git 忽略的**未跟踪文件**（如 `src/partymimic/g1mj_ythn/`、`pipeline/`、`pm`、`deploy` 新增文件）按相对路径拷贝并附 sha256（排除 `.venv`、`results/`、`__pycache__`、>8 MiB 的文件；被跳过的会在 MANIFEST 里列出哈希）。
- `config/`（冻结的训练/评估/部署配置、`args.json`、该舞蹈自己的训练脚本）、`external/`（git 忽略但流水线要 import 的 `results/p0`、`results/p4` 脚本）、`inputs.json`（init checkpoint、源 pkl、npz/csv/padded、`g1_mlp.pt`、`uv.lock`、`pyproject.toml`、最终模型的 sha256）、`env.json` + `pip_freeze.txt`（python/torch/mujoco/mjlab/warp/onnxruntime 版本、GPU+驱动、OS/CPU、sim2real venv）、`timeline.json`（启动/续训/回滚时间线，取自 `supervisor.log`）。
- `MANIFEST.json`：补丁 sha256、三个代码组（`train_tree` / `pipeline_tree` / `deploy_tree`）的 tree 哈希、V1.1 沿用的 `git diff -- src tools` 哈希（`17fcb037a1dd`）、训练源码自首次启动后是否被改过（按 mtime）。
- `REPRODUCE.md`：从干净 checkout 逐步复现的命令，并写明**哪些做不到逐位复现**（GPU 浮点原子累加、CUDA/驱动/GPU 型号、inductor/triton 缓存、并行环境数、评估本身的 GPU 噪声）。
- **`--verify`**：在临时目录用 `git archive HEAD` → 临时 git 基线 → 应用补丁 → 放回未跟踪文件，检查：补丁可应用、重新推导的 diff 哈希与记录一致、未跟踪文件哈希一致、各代码组 tree 哈希一致、重建的 `src` 能 import 并加载该 run 的 robot spec；输出 PASS/FAIL（FAIL 返回码 1）。另给出“当前工作区相对快照是否已漂移”的提示（开发中漂移是正常的，需要时重新 `./pm snapshot`）。
- 快照内容变化时旧快照改名为 `repro_prev_<时间>/` 保留，不覆盖。`./pm run` 的准入记录（`pm/admission.json` → 包内 `ADMISSION.json`）带 `provenance.train_source`（git HEAD、diff 哈希、repro 路径）。

### Sim2Sim 口径（每工况 24 个初态 = 10 个真机站姿 + 14 个采样）
- `eval_bench.sample_inits(n)` 返回 **10 个真机 WalkRun 站姿**（5 个 `ready.json` + 5 个首 tick）**+ n 个插值采样** = n+10 个初态。V1.0/V1.1 文案里的“14 初态”是误写：配置 `sim2sim.n: 14` 是**采样数**，实际每工况 **24** 个初态（sim2sim JSON 的 `summary.n=24` 已核对）。
- 配置 `pipeline/configs/eval/*.yaml` 的 `sim2sim` 段保留 `n`（= sampled，向后兼容），并显式声明 `real_poses: 10 / sampled: 14 / total: 24`；缺省时按 `real_poses=10、sampled=n` 推出，`total` 与二者之和不符时直接报错。
- **CLI `--n` = 采样数，总数 = n + 10**（`./pm sim2sim <舞蹈> --n 6` → 16 个初态）。
- 一致性校验：Sim2Sim gate 会比较 `summary.n`（每工况）与声明的总数，不符则在 gate 结果里写 `warnings`、在日志告警并出现在 `SELECTION_REPORT.md`/`ADMISSION.json`（不改变通过/失败的数值逻辑）。
- 准入记录与部署包现在带：评估器版本（V1.1）、Sim2Sim 初态口径、硬复评 seed/reps/工况数、训练源码 diff 哈希 + repro 快照路径；重新打包时包内 `ADMISSION.json` 另有 `framework_changes_since_previous_package`（与上一版包相比的框架文件差异 + 模型 `policy.onnx` 哈希是否未变；`./pm package <舞蹈> --compare-to <旧包.zip>` 可指定比较对象），`PACKAGE_INFO.md` 同步列出。
- 历史报告（`YTHN_AUTO_REPORT.md`、`SHORT_RUN_GATE.md`、`QUALITY_REPORT.md`）已在文首加更正注记（原文未删）。

### 硬件 profile 与 29 关节审计（`./pm hwaudit <舞蹈>`）
`pipeline/configs/hardware/g1_29dof_rev1_0.yaml` 由官方 Rev 1.0 URDF 生成（含 URDF 哈希与 mode_machine 说明，髋 Roll/膝 = 139 N·m / 20 rad/s，已核实）。`hwaudit` 逐关节对比 URDF、训练(mjlab)、评估器、Sim2Sim、C++ 部署参数，输出 `reports/HARDWARE_AUDIT.md`（按严重度分级的不一致清单 + “待确认硬件后的变更清单”，**不执行**）。已知最重要的两项：部署/Sim2Sim 的髋 Roll 力矩上限 88 N·m（URDF/训练 139），部署髋 Roll 速度 vel_lim 32（URDF 20）。部署资产保持不变。

### 冻结模型的限幅对比（`./pm limits <舞蹈> [--it N] [--limits 88 110 139] [--vel 20]`）
同一 ONNX，不重训；每个变体只替换 `policy_params.txt` **副本**中左右髋 Roll 的 `effort_lim`（可选 `vel_lim`），MuJoCo 部署链闭环（1 参考起点 + 8 扰动起点 × 8 工况，guard 用同一上限钳制）。输出 `reports/limits_it<N>/LIMITS_REPORT.md`。

### 可选的参考动作修复（`./pm repair <舞蹈> [--validate] [--fc-xy 6] [--force]`）
- 只重建**根 XY/Z**：从可靠接触足底质心重建根平移、把触地足压到 z=0；**关节角/角速度、根朝向、帧数、fps 逐位不变**（程序内 assert），原始文件不改。输出 `motions/<舞蹈>_repaired{,_padded}/`。
- `--validate`：对冻结模型做修复前后对比（脚滑、根轨迹与落脚一致性、穿透、悬浮、可靠接触帧上的 CoM/ZMP、闭环、评估网格、训练接触标签 vs MuJoCo 受力接触），生成 `reports/REPAIR_REPORT.md`、`reports/repair/*_repair_compare.mp4`、判据表与是否重训的结论（`RETRAIN_GATE_RECOMMENDED / REPAIR_ONLY_NO_RETRAIN / REPAIR_NOT_TRUSTED`）。
- **读报告注意**：锚定 MPJPE 保留参考根高度，参考悬浮会被计成跟踪误差，修复后它的“下降”不是策略变好；报告同时给出“去除高度偏移”的口径。残余脚滑 = 关节动作本身隐含的滑移（下限）。
- 要用修复数据训练：把训练的动作目录指向 `motions/<舞蹈>_repaired_padded/`（`train_ythn.sh` 读 `PM_MOTION_DIR`），**不要原地覆盖**原始动作；先做 2000 轮短程闸门。

## 13. V1.3：修复数据短程 A/B 对照训练（`./pm abtrain` / `./pm abcompare`）
- `./pm abtrain <舞蹈>`：先给历史成果（原始动作、it40000、部署包、准入记录等 28 个文件）做哈希冻结（`v13/FREEZE.json`），再在 tmux `pm-<舞蹈>-v13` 里**顺序**训练两个独立 2000 轮 run（`runs/ab_repaired_v1`：修复动作；`runs/ab_origctl_v1`：原始动作重复，**运行间噪声对照**）。起点 = 历史运行同一个 `init_from_g1_mlp.pt` 文件，配方 `robust_d_v1`（冻结副本）、seed 0、4096 envs、每 500 轮存盘。run 通过 `run.meta.motion_dir` 指向不同动作目录；默认 run 不变。**不会自动进入长程。**
- `./pm abcompare <舞蹈>`：A=历史运行的 ckpt_002000、B=修复数据、C=噪声对照、另加冻结 it40000 作量级参考；全部在**公共（修复）参考**上、同一批种子化初态、同一部署控制链（保留部署髋 Roll 88 N·m）评估；输出 `v13/V13_AB_REPORT.md`（预登记判据 + GO/NO-GO）、`v13/ythn_v13_compare.mp4`、`v13/ab_result.json`；末尾校验冻结文件未变。
- **判据在训练前写进 `abtest.py`**：6 项实际动力学指标（实际足滑、穿透、CoP 余量、髋Roll 峰值力矩、护栏改写、根轨迹误差）中 ≥3 项显著改善（优于 A 与 C 两者，且 ≥ max(10%、绝对阈值、A–C 差距)），无任何动力学/还原度指标显著退化，鲁棒性不退化 → GO。接触一致率只作参考（与训练目标同源）；高度对齐 MPJPE 去除参考悬浮偏移。
- 读报告注意：足滑有两种定义（足底质心速度 vs 评估器的接触点速度），双支撑更多的步态会把前者放大；报告 §5b 给出两者并说明结论对此敏感。

## 14. V1.4：多舞蹈扫描与泛化验证（`./pm scan`）
- `./pm scan [<舞蹈>…]`：对数据集根目录下所有带 `robot_motion.pkl` 的舞蹈依次 `prepare` + 参考质量分析，输出 `results/_scan/V14_SCAN_REPORT.md`（难度排名、数据缺陷、训练推荐）、`scan.json`、`selection.json`、`scan_ranking.png`。难度 = 10 项指标（单脚支撑/腾空/换脚频率、步幅、质心加速度、ZMP 偏移、参考脚滑、关节速度/加速度、根轨迹与落脚一致性）在被扫描集合内 min-max 归一后的加权平均，并用按排名的同权重复合作稳健性检查。
- 选择规则（确定性）：排除已封存/历史对象（YTHN、SSB）、QA 非 PASS、关节速度超硬件限制、静立缺失、时长 >12 s（C++ 默认运行窗口，更长需显式未验证覆盖）后取最低难度。**默认原始参考数据 + `robust_d_v1`**（V1.3：修复数据 NO-GO，扫描只记录缺陷，不自动修复）。
- 之后 `./pm run <舞蹈> --iters 40000`：2000 轮短程闸门（通过自动续训，失败停训）→ 评估 → 平衡优先选模 → ONNX → Sim2Sim → 打包 → 训练分析 → 质量报告（闭环回放视频）。
- **足滑标准口径（V1.4）**：`quality.contact_point_slip`＝接触点（足底胶囊最低端点，间隙<5 mm，连续两采样，速度<3 m/s）的水平速度，与训练评估器定义一致；`QUALITY_REPORT.md` 与参考分析均输出。V1.3 的 `foot_slip_mps`（足底质心速度）保留在 `abtest.py`，仅作历史口径，双支撑更多的步态会被它放大。

## 15. V1.5：长舞蹈（>12 s）与分阶段闸门
- **闸门修复**：`./pm run` 不再“全程 40000 轮同时训练、旁路检查闸门”。现为两阶段：阶段 1 只训练到闸门轮次（2000，supervisor `--marker gate_stage_done`），在该 checkpoint 上评估并套用 `short_gate`；**仅 PASS 才从 `last.pt` 自动续训到目标轮次**；FAIL 写 `gate_fail` 并停止（不再多花一轮）；评估缺失/超时按 FAIL 处理，绝不默认放行。已通过闸门的 run 重跑时直接续训。
- **分时段跟踪分析**：`python -m pmpipe.segments <舞蹈>`（PYTHONPATH 同 `./pm`）读取 `reports/quality/traj/*.npz`，输出 `reports/SEGMENT_REPORT.md` / `quality/segments.json`：完整播放核对、5 s 窗锚定/全局 MPJPE、根 XY 累计偏差与增长、策略滑移累计、前后半程、护栏/饱和/穿透。

## 16. V1.5.1：部署热修复（DDS 崩溃 / 解压权限 / 默认完整舞蹈 / 一次性授权 / 预热 runner）
来源：DIY V1.5 首次真机触发的 `logs.zip`：模型预热成功，**CHECKING 阶段 1.3 s 后** `EntityDelegate::prevent_callbacks(): Assertion 'false' failed`（SIGABRT，runner rc 134，`CRASH_NO_REPORT`，服务回 IDLE，机器人未被接管）。冻结 V1.5 it40000 / ONNX / 动作数据，只改部署框架与打包器；新部署配置 `cpp_g1_v2`（`pipeline/configs/deploy/cpp_g1_v2.yaml`，v1 不变），`./pm package <舞蹈> --deploy-cfg cpp_g1_v2`（新 run 默认即 v2）。

**1. DDS 崩溃根因**：CycloneDDS-C++ 0.10.2 `prevent_callbacks()` 在 **销毁带 data_available 监听器的 DataReader 时恰有回调在途** 会 `assert(false)`（回调线程持有最后一个引用，自己跑析构）。旧 `Link::measure_traffic()` 每次 F1+F3 和 L1+A 都创建并销毁 3 个临时 reader（`rt/lowcmd` ~500 Hz，回调必然在途），`discover_writers()` 还创建/删除临时 participant。`dev/dds_stress --mechanism` 用慢回调 + 创建/销毁复现了同一断言（rc 134）。
修复（`link.hpp` 文件头写明规则）：**进程内任何 DDS 实体都不销毁**——常驻监视 reader（rt/lowcmd、rt/arm_sdk、rt/user_lowcmd）+ 常驻内置发现 reader（读而不取）+ 10 s 滚动历史（10 ms 采样）；回调只持有 shared 的 `Core`，不引用 `Link`；`Link::close()/~Link()` 只“停用回调、实体寄存到进程生命周期墓地”，进程仍以 `_exit` 结束。重复初始化/释放压力测试：2×600 次流量扫描 + 60 次整 Link 创建/关闭/析构循环对着 500 Hz 热流量无崩溃（`./dev/exit_paths.sh <build> dds`）。

**2. 解压即可执行**：`package.write_zip()` 给每个条目显式 Unix 模式（`create_system=3` + 文件类型位 + 脚本 0755 / 其它 0644 / 目录条目），打包后 `verify_zip_permissions()` 用 `unzip`、`bsdtar` 各解压一次并检查每个 `*.sh` 可执行，失败则**拒绝出包**；zipfile 等忽略权限位的工具解压后，`bash start.sh` / `bash build.sh` 会给同目录脚本补 +x。

**3. start.sh**：默认完整舞蹈（`dance`），取消交互式测试类型菜单（`--mode` / `--list` 保留）。未验证包络的授权改为**一次性现场授权**：第一次键入 `I-ACCEPT-UNVALIDATED`，二进制 `partymimic authz grant` 写 `auth/unvalidated_<mode>.auth`（绑定 policy/params/motion/site.conf/modes.conf/**可执行文件** 哈希、模式、参考模式、窗口、authority、主机；默认 24 h，≤168 h）；之后 `authz check` 通过即不再询问；换包/重编译/改配置/换模式/过期/换主机都会重新询问；`./start.sh --revoke` 撤销，`--authorize` 强制重问，`--no-remember` 每次都问。短语在二进制里校验，site.conf 不能授予授权；无有效授权仍是 NOT ARMED。

**4. 预热 runner（缩短 F1+F3 → 动作的延迟）**：服务启动与每次运行结束后自动起一个 `runner --standby`：硬性拒绝、参数/动作加载、DDS Link+监视、ONNX 加载、8 次只读推理预热、全量 sha256 都已完成，**不持锁、无命令 writer、无 RPC**；接管前 5 s lowstate 预滚在待机期间由私有环缓冲保存，触发时并入日志。F1+F3 只写触发文件，随后**仍执行全部实时安全检查**（最近 1 s 滚动历史的状态流/写者判别、GetFsmId、guardian 心跳、锁/急停、姿态、冻结哈希**重新全量计算并与启动时比对**）→ 准备就绪；L1+A 后的 GO 复核、ARM 预检（前向 0.3 s 窗口）、接管、退出流程不变；每次运行仍是全新进程。实测（回环）F1+F3→READY：预热 **0.12–0.14 s**，冷启动 **1.6 s**（真机模型加载更慢，差距更大）。预热 runner 崩溃 → 日志记录、退避重启（3 次快速失败后停用，F1+F3 回退冷启动）；`--no-standby` 关闭。

**5. 保留的保护与退出行为**：F1+F3 / L1+A / 双击 START / L2+B、guardian、所有安全门、88 N·m 限幅、急停 Passive 路径均未改；正常/可恢复异常退出 → SETTLE → IDLE（服务继续运行，可再次触发，预热 runner 随即重启）；不可恢复故障（倒地、紧急被动、WalkRun 恢复失败等）仍 HOLD 需 `./ack.sh`。

**6. 计时**：`plan_timeline()`（policy.cpp）单一函数规划 前导冻结 + 整段动作 + 末帧保持，runner、selfcheck 7k、打包器（`timeline_check`）共用：dance 窗口必须到达最后一帧、runner 保险时限须比规划总时长多 ≥0.5 s，否则拒绝出包 / 拒绝运行。DIY（2271 帧 = 45.42 s）：前导 1.0 s（50 帧）+ 窗口 45.5 s → 参考到达第 2320 帧（最后一帧）→ 末帧保持 0.5 s，规划 46.88 s，保险 48.0 s（余量 1.12 s）；回环实测 2346 个控制 tick、末帧保持 27 tick。

**测试**：`./pm-test`（42 项，含 zip 权限/时间线/start.sh/配置）；`partymimic selfcheck`（22 项，新增 7k 时间线、7l 一次性授权）；`deploy/partymimic_cpp/dev/exit_paths.sh <build_dir> [scenario…]` 新增场景 `standby` `dds` `timeline` `authz_once`（回环机器人，不连真机）；`PM_ROOT_OVERRIDE=<解压后的包> TL_RUN_S=45.5 … timeline` 可用目标舞蹈的包跑整段回放。

## 17. V1.6 B1：接触识别 / 地面标定 / 根XYZ+腿IK 修正 / 数据 QA 门（`./pm contact-refine`、`./pm contact-qa`）
- 模块 `pipeline/pmpipe/contact_refine/`（与舞蹈无关）。`./pm contact-refine <舞蹈> --analyze-only` 只诊断原始 50 Hz npz；不带该选项则输出 `results/<舞蹈>/motions/<舞蹈>_contactfix{,_padded}/*.npz`（与 prepare 产物格式一致，可作 `PM_MOTION_DIR`；body 状态/速度用 prepare 同一转换器重算；附 `contactfix_info` 元数据：版本/参数/原始 sha256），以及 `results/<舞蹈>/v16/B_contact/`（`qa_verdict.json`、`metrics.json`、`CONTACT_REFINE_REPORT.md`、逐帧 CSV、图；`--video` 渲染原始 vs 修正对照视频）。
- 只改根平移 XYZ 与 12 个腿关节（≤6°/踝≤10°）；上肢/腰/头、根朝向、帧数、fps 不变。腾空仅在质心满足弹道（加速度≈−g）时承认，其余“悬浮”不算腾空。
- **数据 QA 门**：23 条判据写死在 `contact_refine/config.py::QA_GATES`（有 sha256 冻结测试）；`./pm contact-qa <舞蹈>` 独立重判已有修正文件，退出码 0=PASS / 1=FAIL。B2 训练前应读取 `qa_verdict.json`，FAIL 则不得用该修正数据训练。
- 结论与局限见 `results/diy/v16/B_contact/CONTACT_REFINE_REPORT.md`（DIY PASS；ythn/kun 盲测 FAIL，lafanjumps 含真跳跃 PASS）。

## 17. V1.6 B 线：基于已训练权重的微调对比（`./pm finetune` / `ftstatus` / `ftcompare` / `v16-export`）
- `./pm finetune <舞蹈> --from-ckpt 40000 --motion-dir <padded 目录> --run-id <名> --iters N [--seed S] [--lr X] [--mode resume|init] [--after <run>]`：把起点 run 的**完整状态** `last.pt`（`ckpt_*.pt` 只含权重，不能续训）复制进新 run（源 run 只读），resume 式微调（保留优化器/EMA/迭代数/lr 调度；`--mode init` 仅载权重、重置优化器），`--iters` = 追加轮数；tmux 监督器（崩溃/NaN 回滚）+ 评估守护（在该 run 自己的参考上评）+ GPU 监控；写 `pm/finetune.json`、冻结训练配置、repro 快照。运行的参考动作写进 run meta（`Run.motion_padded_dir/motion_padded_npz/motion_npz`），硬复评/导出/Sim2Sim/打包都用它而不是舞蹈的原始动作。
- `./pm ftcompare <舞蹈> --a <对照run> --b <实验run> [--a-extra …] [--b-extra …]`：协议预登记于 `results/<舞蹈>/v16/B_train/FT_PROTOCOL.md`（2×2 参考矩阵、第三方指标、显著性、G0–G4 GO/NO-GO）；全部结果缓存于 `v16/B_train/ftc/`；报告 `FT_COMPARE_REPORT.md`。`rollout_batch.py` 每个 rollout 重建仿真模型（避免 `get_model` 缓存被摩擦工况污染——冒烟测试中发现并修复，已验证逐位可复现）。
- `./pm v16-export <舞蹈> --run-id <实验run>`：仅当 ftcompare=GO、该 run 属于实验组且 ckpt 哈希未变、其参考带 `qa_verdict.json`=PASS 时，才对**被验收的那个 checkpoint**做硬复评→ONNX→Sim2Sim→（准入 PASS 才）C++ 包（默认 deploy cfg `cpp_g1_v3`，无则 v2）；否则返回码 5 拒绝，无人工放行开关。
