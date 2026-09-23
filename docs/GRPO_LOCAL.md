# GRPO 接入、奖励与数据契约

双V100、Qwen3-8B和API裁判的集群操作见[集群快速开始](CLUSTER_QUICKSTART.md)。本文说明奖励定义、数据契约和失败处理。

当前复用 **TRL 0.26.2 `GRPOTrainer` + PEFT 0.18.0**。项目仅实现任务数据、环境执行和裁判奖励适配，采样、分组优势、KL、反向传播和优化器由 TRL 负责；没有另写 GRPO 算法或分布式训练框架。

实现依据为已安装版本的 `trl/trainer/grpo_trainer.py`、`grpo_config.py`，框架入口见 [TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.26.2/en/grpo_trainer)，算法来源为 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。

## 优化单位

训练单位是“固定可见状态下的一次动作”。模型生成 JSON 后环境实际执行，裁判比较该动作及工具返回与固定参照动作。参照动作不进入策略提示，也不是金标准：策略可以被判得比参照更好。完整12轮任务由 Agent CLI 执行，用于后续闭环评测。

**尚未实现整段多轮轨迹 GRPO。** 当前 Transformers 4.57.3 不满足 TRL 0.26.2 原生 `tools` 接口要求的 Transformers ≥5；该版本自定义 `rollout_func` 的接入分支依赖 vLLM。没有把环境返回当成模型生成 token 训练。

## 奖励定义

版本 `agent-next-action-pairwise-v1`：

- 策略输出无法解析或行动被环境拒绝：`-1`，作为模型实际错误。
- 合法行动：`0.8 × 成对裁判结果 + 0.2 × 状态推进`；成对结果取胜1、平0.5、负0，权重来自 `REWARD_SPEC`，代码不另写字面量。
- 每个样本做 AB/BA 两次调用。展示顺序换位后判决不一致，视为顺序伪影，**不可用**。
- 策略复现参照动作时两个候选逐字节相同，任何胜负都只是顺序伪影，直接判平局且不发起裁判调用。
- 裁判调用失败、解析失败或维度证据不足：抛出 `UnusableReward` 中止该批，**不转成零**。TRL原生奖励聚合使用 `nansum`，所以适配器在进入聚合前检查。
- 同组奖励全部相等：该组优势为零，是一次空更新而不是脏数据。记入 `zero_advantage_groups` 并继续，不虚造差异，也不因简单检查点而卡死训练。
- 组内必须是同一状态与参照。DDP 下单个 rank 只持有部分组，跨进程 `all_gather_object` 汇总后再校验组成。
- 任一 rank 奖励不可用时所有 rank 同时中止，避免一边反向传播、一边已退出。已完成的先前批次不回滚，失败日志保留。

## 数据契约

真实入口需要显式 `--suite`，目录包含 `train.json`、`eval.json` 和 `freeze.json`。仓库已提供 `experiments/agent_v1/suite`；也可用 `scripts/build_agent_suite.py --output <dir>` 重新生成。

每条记录字段为 `id, split, exposure, family, source, prefix, anchor`：`prefix` 是可回放的先前合法动作JSON字符串列表，`anchor` 是本状态可执行的参照动作。只从prefix重建策略可见输入，参照与评分不进入提示。

套件由真实环境上的 BFS 枚举生成，因此检查点不可能编码不可达状态；按可见观测签名去重，参照动作优先选择能推进状态的合法动作，避免参照是原地不动。评测检查点按深度分层各取一个，不按数量比例抽样——状态树越深分支越多，比例抽样会让评测集几乎全落在最深一层。

`freeze.json` 绑定两份数据文件的哈希、`task_id=sealed-delivery-dev-v1`、`evaluation_scope` 和 `reward_spec_sha256`。数据文件或奖励定义任一改动都会使套件失效。当前只有一个设计任务族，允许明确为 `development_only` 的训练/开发检查点划分；不能伪造独立留出，`independent_holdout` 下同族会被拒绝。

`characore/grpo_data.py` 另外拒绝：重复ID、split字段与目录不符、开发检查点被改标为留出、同族泄漏、可见检查点重复、参照动作在该状态不可执行。

## 裁判后端

`--judge-backend` 三选一：

- `api`：真实远程裁判，密钥从 `.env` 读取，必须显式 `--allow-api`。
- `stub`：`characore/stub_judge.py`，离线、确定性，只比较环境执行是否成功。用于验证请求构造、严格解析、AB/BA 一致性和奖励装配，**不是质量信号**，其产物不能当作角色一致性结果。
- `local`：本地裁判模型，需 `--judge-base`，固定在 CPU、greedy、seed17、关闭thinking、输出1024 token上限。

超长输入拒绝，不静默截断。仅 `messages` 交给模型，映射与预期不进入推理输入。保存模型及提示绑定、原始响应、每次解析失败、重试与token数。

裁判协议为 `judge-v0.3`，严格校验：四维 C/S/A/Q 状态与分数、证据引用必须在允许ID集合内、胜负必须给出成对证据引用、不允许重复JSON键。本任务每轮都必须行动，因此 A 维不允许 `not_applicable`。

## 可运行命令

标准库环境：

```powershell
python scripts/build_agent_suite.py --output runs/agent_suite_01
python -m unittest discover -s tests -v
```

训练环境（模型路径是占位值，不会自动下载）：

```powershell
python -B scripts/train_grpo.py --tiny --steps 2 --output runs/tiny_grpo_01
python -B scripts/train_grpo.py --preflight-only --base model_cache/Qwen3-8B --suite experiments/agent_v1/suite --judge-backend stub --output runs/preflight_01
python -B scripts/train_grpo.py --base model_cache/Qwen3-8B --suite experiments/agent_v1/suite --judge-backend stub --device cuda --precision fp16 --steps 2 --output runs/grpo_stub_01
```

`--preflight-only` 不请求API、不加载模型、不更新参数。`--tiny` 使用 CPU 随机微型模型和合成 token-ID 均值奖励，验证有限非零梯度、LoRA 参数变化、冻结参考不变、保存重载一致；**不是裁判奖励训练，也不是角色效果结果**。

## 产物与限度

`verification.json` 给出优化步数、LoRA参数变化、梯度有限非零、冻结参考未变、适配器重载一致、DDP跨rank参数一致。`before_development_eval.json` 与 `after_development_eval.json` 给出训练前后在评测检查点上的greedy动作及环境执行结果。

训练前后的开发检查点比较不能代替整段任务评测或独立泛化比较。裁判与人工判断的一致率尚未测量，因此奖励可靠性未经校准，`manifest.json` 中据实记为 `uncalibrated`。Windows应用控制阻止bitsandbytes DLL（WinError4551），当前不能认定4-bit路径可用；CPU微型更新不受影响。
