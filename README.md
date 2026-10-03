# CharaCore

用 GRPO + LLM 裁判（RLAIF）训练小模型保持两个角色各自的说话风格。给定角色设定卡、一个情境和对方的一句话，策略模型（Qwen3-4B + LoRA）输出该角色的一句回复；奖励来自裁判对“策略回复 vs 冻结基座回复”的 AB/BA 成对比较，加规则风格分与反投机惩罚。

同人研究，非商业用途。凌波丽、明日香及《新世纪福音战士》相关设定的版权归原权利人（khara、GAINAX 等）所有。仓库中的角色设定卡是原创撰写的性格概述，情境与台词均为原创，不包含原作台词。

## 文档

- [设计](docs/DESIGN.md)：任务、数据、奖励、被拒组掩码、训练证明、评测与限度。
- [集群快速开始](docs/CLUSTER_QUICKSTART.md)：从传代码到训练、评测的完整单行命令。
- [进度](docs/PROGRESS.md)：当前状态、结果与奖励投机记录。

## 要点

- 奖励不可用（调用失败、解析失败、AB/BA 不一致）时整组拒绝，在训练器中掩掉该组的损失并计数，从不用 0 顶替。
- 裁判响应严格 schema 校验；同批相同请求只调用一次并复用。
- 硬违规（出戏、长度越界、口头禅堆砌、冒充另一角色）直接 −1，不调用裁判。
- 数据按情境模板划分训练 / 测试，SHA-256 冻结；基座回复只生成一次并哈希绑定。

## 本地

```powershell
python -m unittest discover -s tests -v
python scripts/build_style_suite.py --output runs/suite_check_01
```

单元测试只需标准库（有 torch 时多跑一项掩码测试）。训练依赖见 `requirements.txt`；训练与评测在 GPU 集群执行。

## 目录

- characore/：角色设定、奖励、裁判协议、训练器子类与数据加载。
- scripts/：套件生成、基座回复、survey、裁判一致性检查、训练、评测。
- experiments/：冻结的数据套件与基座回复。
- tests/：单元测试。

旧的“单步动作一致性”方案保留在 `archive/action-consistency-v1` 分支。代码采用 [MIT License](LICENSE)，许可不覆盖第三方角色与作品。
