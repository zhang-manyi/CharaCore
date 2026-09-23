# 项目进度与交接

更新时间：2026-09-23。当前范围以 [DESIGN.md](DESIGN.md) 为准。

## 当前阶段

**主线是单步动作 GRPO：环境、奖励、训练入口和训练数据均已就绪，克隆后可直接训练。** 原创角色“岚”的密封档案递送是设计测试环境。真实 Qwen3-8B 的 GRPO 训练在 GPU 集群执行，本地只做机制验证。

| 项目 | 实际状态 |
| --- | --- |
| Agent 环境 | 单角色短任务、三类本地工具、显式状态、事件记忆、JSON 轨迹可校验回放；`verify` 三条脚本轨迹通过 |
| 训练数据 | `experiments/agent_v1/suite` 冻结：71 训练 / 4 评测检查点，BFS 枚举真实环境生成，评测集覆盖深度 1–4 |
| 奖励 | `agent-next-action-pairwise-v1`：非法动作 -1；合法动作 0.8×成对裁判 + 0.2×状态推进；AB/BA 顺序校验；不可用即中止，不补零 |
| 裁判 | 协议 `judge-v0.3` 严格解析；三后端 api / stub / local。与人工一致率未测量，可靠性未校准 |
| 训练入口 | TRL 0.26.2 GRPOTrainer + PEFT LoRA；CPU 随机微型模型 2 步更新此前通过（单进程与双进程 Gloo/FileStore）。本机现已无法执行 TRL，见「具体阻塞」 |
| 模型与资源 | 集群策略模型 ModelScope `Qwen/Qwen3-8B`，双 V100 FP16 基座 + FP32 LoRA。单卡 32GB 实际峰值待实测 |
| 效果结论 | **无。** 尚无真实任务训练前后的效果改善结论 |

## 下一步最小任务

1. 在集群上按[集群快速开始](CLUSTER_QUICKSTART.md)第 6 节用桩裁判跑通完整训练路径，确认显存与吞吐，再按第 7 节换真实 API 裁判出一次 GRPO 结果。
2. 保留训练前后 `before/after_development_eval.json`，从实际动作与执行结果诊断策略变化；报告原始计数，不据单次开发运行声称能力改善。
3. 需要偏好可靠性时再补本任务的人工标注与一致率测量；当前据实记为 `uncalibrated`，不以示例配置冒充校准通过。

## 具体阻塞

- **离线闭环：** 已交付，无运行阻塞；纯脚本与回放仅依赖标准库。
- **单步动作 GRPO：** 框架接入、奖励失败处理、数据契约、套件生成与微型更新均已验证。剩余未知量是 V100 显存/吞吐和真实 API 的连通性与返回质量。
- **整段多轮轨迹 GRPO：** 未实现。Transformers 4.57.3 不满足 TRL 0.26.2 原生 `tools` 接口要求的 ≥5；自定义 `rollout_func` 分支依赖 vLLM。
- **独立泛化比较：** 评测检查点与训练检查点同属一个原创任务族，是开发集而非盲测。正式泛化比较需要独立来源分组和未参与开发的留出。
- **4-bit 路径：** Windows 应用控制阻止 bitsandbytes DLL（WinError4551），不能认定可用。
- **本机不能再运行 TRL：** 同一应用控制现在也阻止 pyarrow 的 `_csv` DLL，而 `datasets` 依赖它，因此 `import trl.trainer.grpo_trainer` 在本机失败。这是系统安全策略，未绕过。受影响的只有本机执行：`--tiny` 与真实训练都需在集群运行。不依赖 TRL 的部分仍在本机验证（见下）。

边界：代码默认不请求 API，须显式 `--allow-api`；不上传权重或密钥；模型一律本地离线加载，不自动下载；输出目录排他创建，不覆盖既有结果。

## 当前产物入口

- [集群快速开始](CLUSTER_QUICKSTART.md)、[密钥配置模板](../.env.example)、[API连通性检查](../scripts/check_judge_api.py)。
- [GRPO 接入说明](GRPO_LOCAL.md)、[训练入口](../scripts/train_grpo.py)、[套件生成](../scripts/build_agent_suite.py)、[评测计划](../experiments/agent_v1/evaluation_plan.json)。
- [运行入口](../scripts/run_agent.py)、[任务环境](../characore/agent.py)、[本地推理](../characore/local_policy.py)、[奖励](../characore/grpo_rewards.py)、[跨进程奖励同步](../characore/distributed_rewards.py)。
- 测试：[机制测试](../tests/test_agent.py)、[奖励与数据门禁](../tests/test_grpo.py)、[API与集群路径](../tests/test_api_cluster.py)。

## 历史记录

原神来源审核、Stage B 校准材料（36对/72请求，人工 0，`calibration_passed=false`）、DPO 训练器及相关文档保留在 `genshin-data-audit` 分支，不在主线。当时的结论未改变：那批材料不能授权本原创任务的奖励，正式 train/test 仍为 0/0。主线改为直接用本任务自己的开发检查点做单步动作 GRPO，不再依赖那条数据线。

2026-09-23：主线收敛为强化学习一条线。移除原神子系统后剩余 29 项单元测试通过。奖励安全机制保留（不补零、跨 rank 同时中止、冻结参考校验、输出目录排他创建）；原先要求人工校准审批才能训练的硬门禁降级为据实报告 `uncalibrated`，因此训练不再被未完成的人工材料阻塞。同组奖励全等由“中止该批”改为“计入 `zero_advantage_groups` 并继续”——该组优势为零本就是一次空更新，中止会在简单检查点上卡死训练。

2026-09-23 交付核验（在 `main` 的全新克隆上执行，非当前工作区）：

| 检查 | 结果 |
| --- | --- |
| 29 项单元测试 | 通过，仅需标准库 |
| `run_agent.py verify` | 三条脚本轨迹通过，仅需标准库 |
| 套件随仓库可用 | `experiments/agent_v1/suite` 存在；重新生成的 `files` 与 `reward_spec_sha256` 与冻结值逐一相同，生成过程确定 |
| `--preflight-only` | `passed=true`，未加载模型、未请求 API |
| 奖励全链路（桩裁判） | 在真实检查点上跑 14 组：非法动作与不可解析输出得 -1；复现参照动作判平局且裁判调用数为 0；需要真实判决的 8 组各发起 AB/BA 两次调用（共32次），严格解析全部 `ok`，`rewards.json` 如实记录 `zero_advantage_groups` |
| GRPOConfig 契约 | 入口传入的 31 个参数在 TRL 0.26.2 中全部存在，构造成功 |
| 奖励调用契约 | TRL 以 `completions=` 加 `**reward_kwargs`（来自数据集列）调用奖励函数，与 `ActionReward(completions, prefix, anchor, **kwargs)` 一致；聚合确实使用 `nansum`，故适配器必须在聚合前拦截不可用奖励 |

未在本机验证：`--tiny` 与真实 GRPO 训练步（pyarrow DLL 被系统阻止）、V100 显存与吞吐、真实 API 裁判的连通性与返回质量。这四项在集群首次运行时确认。

2026-09-22 及更早的原神数据审核、校准包与 DPO 记录见 `genshin-data-audit` 分支的同名文件。
