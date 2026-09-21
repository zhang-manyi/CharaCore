# 数据选择与准备

当前路线见 [DESIGN.md](DESIGN.md)，执行状态见 [PROGRESS.md](PROGRESS.md)。

## 选择目标

优先选择现成 RPG、ACG 或游戏角色数据；不限定作品，不要求预先标记为强化学习用途。重点是角色依据、当前情境和相关历史能否支持合理性判断。

优先复用可信偏好对。若只有情境或 SFT 对话，复用其输入和参考材料，采样与评分后形成 DPO 偏好，同时支持 GRPO 在线采样；原回答不自动视为优选答案。

初期参考规模为数百训练情境、约百独立评测情境，按可用数据与预算调整，先做小试运行。候选、改写和多角色视角不按独立场景重复计数。

## 有限选型检查

- 查看少量真实记录，确认角色、上下文、行动/台词和来源字段可用。
- 区分作品摘录、整理注释、生成对话、人工/模型偏好标签。
- 核对语言、数据许可及底层材料边界；记录 URL、版本、哈希和必要署名。
- 核对上游 train/test，按场景及变体分组去重；人物资料可能含未来信息，不能直接全量进入提示。
- 核验相关评价研究的任务与协议，再决定是否适用，不预设统一权威标准。

选型到足以决定能否复用即可，不做镜头级来源工程，不扩大手写剧情或审阅表。原始缓存和新运行分别放入被忽略跟踪的 data/raw/ 与 runs/。

## 现有 DPO 输入契约

真实训练通过 --suite 指定目录，没有内置默认数据集。目录包含以下 JSON 文件：

| 文件 | 作用 |
| --- | --- |
| profiles.json | 角色 ID 到 name、role、stable_preferences、style 的映射 |
| train.json | 训练记录列表 |
| eval.json | 与训练情境族隔离的评测记录列表 |
| evaluation_plan.json | 版本化评测计划；具体新评测协议尚待实现 |
| freeze.json | files 字段记录 profiles.json、eval.json 和 evaluation_plan.json 等冻结文件的 SHA-256 |
| train_manifest.json | evaluation_freeze_sha256 与 train_sha256 绑定训练数据和冻结版本 |

每条记录包含 id、family、character、goal、relationship、history、emotion、situation、user、candidates（a/b）、preference、evidence。preference 可为 a、b、tie 或 insufficient；只有 a/b 进入严格 DPO 训练。标签、候选和 evidence 不进入策略提示。

这是当前训练器支持的接口，不是正式数据来源或最终 GRPO 输入协议。选定数据后增加最小适配；保留 source_id、上游划分、场景族、来源版本和处理记录。不能用缺少有效分组的 ID 随机切分来代替场景隔离。

先冻结 train/dev/test 分组，再生成训练候选；dev 用于校准和配置选择。当前读取器只加载 train/eval，dev 管理与新裁判协议需要适配阶段补齐。保留平局、证据不足和过滤原因，不为凑规模降低要求。

## 完成标准

产出一份简短的数据选择报告：实际样例与定位、选择理由、可用情境规模估计、使用条件、划分风险及最小适配方案。满足后转入评分与裁判校准，不无限延伸考据。
