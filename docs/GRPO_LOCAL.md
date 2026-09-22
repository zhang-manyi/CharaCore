# 本地 GRPO 接入与校准使用

双V100、Qwen3-8B和API裁判的新入口见[集群快速开始](CLUSTER_QUICKSTART.md)。下文保留初版本地裁判验证；新增API后可用`--judge-backend api --allow-api`替代本地`--judge-base`，密钥从`.env`读取。

当前复用 **TRL 0.26.2 `GRPOTrainer` + PEFT 0.18.0**。项目仅实现任务数据、环境执行和裁判奖励适配，采样、分组优势、KL、反向传播和优化器由 TRL 负责；没有另写 GRPO 算法或分布式训练框架。

实现依据为已安装版本的 `trl/trainer/grpo_trainer.py`、`grpo_config.py`，框架入口见 [TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.26.2/en/grpo_trainer)，算法来源为 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。当前单机小试沿用已验证环境，不引入需要另一套资源与部署的训练栈。

## 已验证的范围

- 随机微型模型在 CPU 上实际执行2步 GRPO，4个同提示样本一组，使用合成 token-ID 均值奖励。验证有限非零梯度、LoRA 参数变化、冻结参考不变、保存重载一致；**不是 LLM 裁判奖励训练，也不是角色效果结果**。
- 真实任务入口的训练单位为“固定可见状态下的一次动作”。模型生成 JSON 后，环境实际执行，裁判比较该动作及工具返回与固定参照动作。参照动作不进入策略提示，不自动当作正确答案。完整12轮任务仍由已有 Agent CLI 执行，用于后续闭环评测。
- **尚未实现整段多轮轨迹 GRPO。** 当前 Transformers 4.57.3 不满足 TRL 0.26.2 原生 `tools` 接口要求的 Transformers ≥5；该版本自定义 `rollout_func` 的接入分支依赖 vLLM。没有把环境返回当成模型生成 token 训练，也没有为演示升级外部环境。
- 原神36对材料原样保留。新增校准执行与统计入口，不新增刺激、不代填人工。现有包校准即使通过，也不能授权原创档案任务的奖励。

## 可运行命令

标准库环境可先计算当前待校准状态；输出目录须不存在：

```powershell
python scripts/calibrate_judge.py report --packet runs/stage_b_20260922/calibration_v03_02 --output runs/judge_report_02
python -m unittest discover -s tests -v
```

以下命令须使用已有兼容训练环境的 Python。模型路径是占位值，不会自动下载；删除 `--limit 2` 才会按原计划混排执行72个请求，每个最多重试一次：

```powershell
python -B scripts/calibrate_judge.py run --packet runs/stage_b_20260922/calibration_v03_02 --base model_cache/Qwen3-1.7B --limit 2 --output runs/judge_smoke_02
python -B scripts/train_grpo.py --tiny --steps 2 --output runs/tiny_grpo_02
```

裁判默认 CUDA、greedy、seed17、关闭thinking、输入4096/输出1024 token上限；可用 `--device cpu`。超长输入拒绝，不静默截断。仅 `messages` 交给模型，映射、预期和人工标签不进入推理输入。保存模型及提示绑定、原始响应、每次解析失败、重试与token数。

本轮实际结果在 `runs/grpo_20260922_01/`：

| 产物 | 实际结果 |
| --- | --- |
| `tiny_grpo_01/verification.json` | 2个优化器步，12个张量更新，24次梯度回调中18次非零；冻结参考不变，重载最大误差0 |
| `tiny_grpo_01/sampled_rewards_*.json` | 框架真实采样的token、文本与合成奖励，不是预写完成结果 |
| `local_judge_smoke_01/calls/` | 原计划中2个请求，各2次尝试，共4次真实本地推理；均缺顶层`reason`，严格解析失败，没有自动补齐 |
| `local_judge_smoke_01/report.json` | 已调用2、未调用70、无人工结果、`calibration_passed=false`；不能用2条推断完整72条表现 |
| `delivery_report.json` | 环境、测试、保护核验及限制汇总 |

环境还报告 bitsandbytes DLL 被 Windows 应用控制阻止（WinError4551）。此次 CPU/非量化微型训练成功；4-bit 路径目前不能据旧记录认定可用，没有绕过系统控制或修改外部环境。

## 人工结果接入

按原[校准流程](GENSHIN_STAGE_B_CALIBRATION.md)，两人各自复制并完成 `human/reviewer_1.json`、`reviewer_2.json`，结果放到新的目录，原包不能编辑。这里的“人”必须是真实复核者，单元测试内的合成表不算人工。

同一结果目录内另外提供 `adjudication.json`：

- `reviewer`、`completed_at`：实际仲裁人和时间。
- `reviewer_files_sha256`：两份首轮文件名到 SHA-256 的映射，锁定首轮后填写。
- `annotations`：全部36个中性id；每项包含 `id`、`review_status`、`winner`、`reason`、`preference_evidence_ids`。未解决的项目保持 `pending`，不会被悄悄剔除来达到通过门槛。
- `model_citation_audit`：实际复查模型理由后填写 `judge_identity`、`results_sha256`、`reviewed_request_ids`（排序的全部请求id）、`completed`、`unsupported_citations`（错误引用数）。前两项来自本次模型报告。只有合法引用ID并不能证明理由有据。

```powershell
python scripts/calibrate_judge.py report --packet runs/stage_b_20260922/calibration_v03_02 --results runs/judge_run_02/calls --human runs/human_results_01 --output runs/judge_calibration_report_02
```

统计包含首次/重试后合法率、AB/BA映射后的顺序一致率、人工全部/可判定/严格胜负/不足子集一致率、两人一致率、位置分布、分维度状态、保守组与诊断类别原始计数。缺少人工、少调用、顺序冲突、模型混用或引用未验收都不能通过。模型原始响应、请求和已锁定人审文件的绑定不匹配直接拒绝。

## 奖励处理与真实训练入口

当前奖励是待人工验收的明确提案，版本 `agent-next-action-pairwise-v1`：

- 策略输出无法解析或行动被环境拒绝：`-1`，作为模型实际错误。
- 合法行动：`0.8 × 成对裁判结果 + 0.2 × 状态推进`；成对结果胜/负/同分为1/0/0.5。AB/BA必须映射一致。状态推进仅计新增已知事实、承诺、移动或递交，重复承诺和查询状态不算推进。
- 技术失败、任一维证据不足、方向冲突、缺失/非有限奖励：保存失败并中止该批更新，**不转成零**。TRL原生奖励聚合使用`nansum`，所以适配器在进入聚合前检查。
- 同组奖励全部相等：中止该批，不虚造差异。组内必须是同一状态与参照；不实施自动重采样服务。已完成的先前批次不回滚，失败日志保留。

真实入口需要显式 `--suite`，包含 `train.json`、`eval.json`、`reward_approval.json` 和 `freeze.json`。每条训练/评测记录字段为 `id, split, exposure, family, source, prefix, anchor`：`prefix` 是可回放的先前合法动作JSON字符串列表，`anchor` 是本状态可执行的参照动作。只从prefix重建策略可见输入，参照、评分和审批不进入提示。

`freeze.json` 固定前三份文件哈希、`task_id=sealed-delivery-dev-v1` 和 `evaluation_scope`。当前只有一个设计任务族，允许明确为 `development_only` 的训练/开发检查点划分；不能伪造独立留出，`independent_holdout` 下同族会被拒绝。`reward_approval.json` 必须由实际复核者填写 `reviewer, completed_at, approved`，并绑定 `reward_spec_sha256` 和实际 `calibration_report_sha256`。代码不会自动生成这些已批准材料。

还必须提供**针对本任务且实际通过的**校准包/结果/人工材料，包的 `calibration_scope` 与task_id绑定；现有原神包不满足。当前没有这套任务校准材料，也没有获验收的训练套件。本轮只完成接入与机制验证，不能宣称真实任务已可开始优化。

具备输入后可先运行 `scripts/train_grpo.py --preflight-only --base ... --judge-base ... --suite ... --packet ... --judge-results ... --human ... --output ...`，通过再移除 `--preflight-only`。真实训练固定裁判在CPU，策略可指定CUDA，避免默认在6 GB显卡同时常驻两个模型；裁判校准必须使用相同配置。推理和更新资源仍需届时实测。训练前后保存开发检查点的greedy动作及环境结果，不能代替整段任务评测或独立泛化比较。
