# 现成数据选型与最小适配

日期：2026-09-21。范围：设计总纲阶段 A；少量实样检查，不是完整数据质量审计或正式评测。

后续调整：明确中文与原神角色优先后，当前主选改为 ChatHaruhi 的绫华、钟离、胡桃，CoSER 保留备选。新增实际数量核查及 GRPO 优先实施路线见 [GENSHIN_PILOT.md](GENSHIN_PILOT.md)。以下保留初次 CoSER 选型依据，不代表当前主源已冻结。

## 选型结论

**推荐主源：CoSER 的 `full/` 原始结构化数据，训练仅取非 OOD 书籍的上游训练剧情。** 首批用已核查的 `Pride and Prejudice` 做 10–20 个情境的适配检查，再决定是否扩书。它具备角色资料、当前情境、多轮对话及剧情坐标，比直接使用 SFT 导出更适合本项目的场景隔离与证据追踪。英语文学角色不是首选的 RPG/ACG 题材，但符合总纲中情境证据与可用性优先的顺序。

ChatHaruhi 保留为中文 ACG 备选。已检查材料没有可靠的成对偏好标签；两者均只能先作为情境与候选来源，原回答不自动成为 chosen。本轮不实施裁判、数据冻结或训练。

## 实际比较

| 候选与实查范围 | 角色依据和情境 | 标签、划分与许可 | 判断 |
| --- | --- | --- | --- |
| CoSER，英语；2 本完整书 JSON、全部 200 条上游 test 的结构统计、SFT 文件前 64 KiB | `character_datasets` 含 profile、剧情经历及台词；`plots[].conversation[]` 含 scenario、角色 thought/motivation、dialogues；保留 book/plot/conversation 定位 | 不是偏好数据；有 `split_plot_index`、OOD 书单和 test；数据卡标 MIT，不能据此认定小说内容均可再分发 | 主源，使用 full 并重建分组；不用混合任务且丢失定位的 SFT 导出直接训练 |
| ChatHaruhi，中文春日及神里绫华；5 个台本文件、3 个完整 JSONL 文件，人工查看其少量记录 | 台本有说话人、关系和行为线索；扁平 query/response 缺少场景 ID 与完整历史；另有明确标为 synthesized 的对话 | 未在所查文件中发现可靠偏好或可沿用的 train/dev/test 分组；README 区分代码 Apache-2.0、数据 CC BY-NC 4.0，并提醒角色版权 | ACG 匹配好，但需恢复原场景、区分生成内容并补齐划分，初版成本高于 CoSER |

## 可定位的实样与核查事实

- **CoSER，《傲慢与偏见》`plots[0].conversation[0]`，Chapter 1：** 班纳特太太请丈夫拜访新邻居，动机是女儿婚事；丈夫以讽刺和拖延回应。第一句公开输入含 `have you heard that Netherfield Park is let at last?`。适合检查目标、夫妻关系和回应方式，未必要求实际行动。原 `message` 混有方括号内心描写与括号动作，不能整段当成公开对话历史。
- **CoSER，上游 `test[0]`：** 《傲慢与偏见》`i_p=70, i_c=0`，父亲谈及柯林斯来信及婚约传闻。已确认 test 与 full 对应记录的 `dialogues` 完全相同；必须留在 test，不能用于裁判提示调参或训练。
- **CoSER，《权力的游戏》`plots[0].conversation[0]`，PROLOGUE：** 守夜人讨论是否撤回，角色资历、权威与危险判断都有依据。剧情 summary 已写出罗伊斯死亡及复活，早于该事件的输入不能含这些结局。该书在上游 OOD 书单中，整书排除训练/dev；文件名也不足以证明内容边界，首个 Aegon profile 含后续剧情，不能无审查复用完整传记。
- **ChatHaruhi，`characters/ayaka/texts/24.txt` 与 `25.txt`：** 两文件字节与 SHA-256 完全相同。情境是久利须的丝绸被抢；绫华既提醒他注意对奉行的言辞，又提出以个人方式帮助追回货物，适合评价身份约束下的行动。`1.txt` 是托马对绫华身份、职责与待人方式的叙述，属于他人陈述，不能一律当作绫华当时亲知信息。
- **ChatHaruhi，春日台本 `Haruhi_07_knn3_to_text__101.txt`、`__120.txt`：** 棒球特训与争胜目标可追踪；不能仅因文件名不同就当独立场景。`all_chat_datas.jsonl` 首两行谈写 Python 和预测未来，字段只有角色、query/response、keywords，不能凭目录名断言是原著摘录。
- **生成内容与证据强度：** 春日 synthesis 的 454 行、绫华 generated 的 198 行全部标 `source=synthesized`；春日 chat 的 556 行未标 source。CoSER 数据卡自称源自文学对话，但实样还包含整理后的情境、人物总结与内心注释；本轮没有逐句对照原著或验证这些注释的生成过程，不能称作全部未经加工的原文或人工金标准。

结构统计是全文件计算；以上内容判断仅来自列出的少量样例：

| 已下载文件 | 剧情位置数 / split_plot_index | 划分点前后对话数 | 其他实测 |
| --- | --- | --- | --- |
| Pride and Prejudice | 77 / 69 | 70 / 8 | 前 69 个剧情均有对话；全书按完整 dialogues 精确比较有 2 个重复副本；1 条上游 test 可精确回指 |
| A Game of Thrones… | 302 / 271 | 267 / 27 | OOD，前段也不能训练；全书有 5 个精确重复副本；4 条上游 test 可精确回指 |
| test/test_set.json | 200 条对话 | id=100，ood=100 | 按 book+i_p 是 199 个剧情组：id=100，ood=99；还没有做跨书近重复归并 |

这里的重复副本数为对话数减不同 dialogues 的数量，不是已判定的独立事件数。ChatHaruhi 三个 JSONL 无整行精确重复，也不能推出其台词或情境无重叠。CoSER 发布树有 773 个 full 路径，数据卡宣称 771 本书；本轮没有调和口径，不以该数估算可用训练量。SFT 缓存首条实际是 Environment 任务，且导出只有 conversations；构建代码还包含多角色视角、内心描写开关及 NSP，导出行数不能当角色训练情境数。

## 使用条件与规模判断

CoSER 的 MIT 是发布者的数据卡声明；当前证据不能替底层小说、版本或整理注释确认全部权利。ChatHaruhi 的数据声明要求署名、非商业使用并遵守 CC BY-NC 4.0 条件，代码许可不替代数据许可和角色版权。初版以本地研究、来源索引和处理方法为交付，原始数据留在 Git 忽略目录；若后续再分发数据或商用，另核对实际内容和许可范围。《傲慢与偏见》原作较早，不等于当前整理版本的全部权利已核清。

**已验证容量：** 当前下载材料中，非 OOD 的训练候选上限为《傲慢与偏见》69 个剧情位置、70 段对话，尚未扣除重复、未来信息、证据不足及 dev 留出；不能宣称已经取得数百独立训练情境。上游 id test 有 100 个剧情组，可作为约百个评测情境的候选，过滤后可能不足。

**待验证估算：** 若后续每本非 OOD 书能保留约 40–70 个情境，约 6–9 本可尝试支持 300 train + 40 dev；这只是容量规划，不是已测得的留存率。先检查首批适配的证据质量，再小批增加有清楚来源条件的书，题材合适时优先幻想文学。不凑配额，不将角色视角或多个回答累计为独立情境；暂不翻译英语材料。

## 最小适配方案（待实施）

1. **确定来源与分组。** 固定版本及文件 SHA-256，读取 `full/`、OOD 书单和 test。训练/dev 仅允许非 OOD 且 `i_p < split_plot_index` 的记录；缺失或矛盾的坐标、划分信息隔离待查，不擅用默认 90%。其余剧情及 OOD 全书不得进入训练/dev，包括它们在 `character_datasets` 中的重复台词和经历。用 book+i_p 建初始组，按相同/重叠台词和同一原事件合并跨 plot、chunk、书名变体；任一成员属于上游留出，整组保留为留出。确认重复位置和系列别名后才冻结。
2. **拆分证据与策略可见输入。** 先建无候选的 `contexts.jsonl`、`sources.json`、`split_manifest.json`、`exclusions.jsonl`；另存隐藏参考和注释，不为满足旧契约伪造 a/b。每个场景选择一个明确的角色和响应截点，其他视角保留同 family。只取截点前该角色可知的公开历史；去除他人内心、未来结局、整段剧情复述和泄露后续的 Environment 概括。目标角色自己的 thought/motivation 也需检查时间范围和证据，未知目标、关系或情绪显式标未知，不凭空补剧情。
3. **再冻结 train/dev/test。** 从上游训练候选按归并后的 family 留出 dev，test 沿用合格的上游 id 情境，OOD 暂另存。所有变体随组走；角色可重合，只主张已知角色的未见情境，并在实际训练角色集合确定后核查 test 覆盖。冻结分组、dev、test、角色依据和版本化评测计划，再生成训练候选。已人工看过的 `test[0]` 登记为审查暴露，不能据此设计裁判特例或声称完全盲测；需要纯未审查子集时将其从主评测排除，不转作 dev。
4. **接入现有协议。** 新增小型 CoSER 适配入口；`characore/protocol.py` 分开校验无标签情境和有偏好训练行，补齐 dev 哈希及三方 family/重复检查。保留 `eval.json` 表示 test，增加 `dev.json`；测试情境不伪造候选或偏好。现有训练器只校验/记录 eval 哈希、未执行正式评测，后续配置和检查点选择仅使用 dev。提示需支持英语，允许自然回答及必要动作，解除现有中文强制一至三句具体行动的限制；语言与输出协议对基座/DPO/GRPO 一致冻结。无动作样本允许行动维度不适用。
5. **最后生成偏好。** 先在 dev 做裁判校准，确认使用条件与调用预算，再从冻结 train 输入采样候选并记录生成器、参数和裁判证据。严格 a/b 才生成旧格式 `train.json`；tie/insufficient 保留但不训练，调用失败独立记录。GRPO 后续复用同一个 train 输入池。不要把隐藏原回答的字面相似度当角色一致性奖励。

| 现有字段 | CoSER 映射与限制 |
| --- | --- |
| id / character / family | 版本+文件+plot+conversation+响应截点；角色用书/系列命名空间；family 使用归并表，不能仅用新 ID |
| profiles | profile 与可见时期的角色描述中经审查的身份、稳定偏好、风格；共享部分不得从 dev/test 的回答或后续经历推导，情境特有信息留在记录中 |
| goal / relationship / emotion / situation | 当前角色可用的目标、关系和情绪证据，加去除不可知信息后的 scenario；不把 plot summary 原样填入 |
| history / user | 带说话人的截点前公开对话和可观察动作；最近触发输入单列 user，目标回答及之后的内容隐藏 |
| candidates / preference / evidence | 下一阶段生成及校准后填入；参考原回答另存。evidence 保留来源指针、证据级别与评分理由，不进入策略提示 |
| 附加溯源 | source_id、原始上游划分、book/i_p/i_c/i_u、文件版本与哈希、合并/过滤/改写记录、参考可见性 |

下一阶段验收以少量真实适配记录为准：能回指原文位置，无上游留出进入 train/dev，无角色未知信息或参考答案泄露，dev/test 变体不跨组，英文提示可表达非行动回答。只在这些接口实际修改后补针对性测试；本轮未修改训练代码，不重跑已通过的训练机制检查。

## 来源与本地产物

- CoSER [数据卡与文件树](https://huggingface.co/datasets/Neph0s/CoSER/tree/7cc80430f92532cda85df45015a4aca8ecc068d0)，revision `7cc80430f92532cda85df45015a4aca8ecc068d0`；[构建代码](https://github.com/Neph0s/CoSER/blob/624efc50db7be3a4e37cec6a1c58ed434dc9f2ce/data_construction/transform.py)，revision `624efc50db7be3a4e37cec6a1c58ed434dc9f2ce`。划分政策同时依据代码、OOD 文件和实样，不宣称代码与当前发布每条记录均已对齐。
- ChatHaruhi [README 与许可说明](https://github.com/LC1332/Chat-Haruhi-Suzumiya/blob/290bf4ad22076156083804013012847a77c0646c/README.md)，revision `290bf4ad22076156083804013012847a77c0646c`；样例均来自该版本，路径见上文及本地来源清单。
- 复用 `data/raw/reuse_audit_20260921/sources.json` 的 16 个缓存，逐个重新校验 SHA-256；新增 `data/raw/selection_20260921/`，3 文件共 7,616,513 字节（约 7.26 MiB），原 URL、时间和完整哈希见 `sources.jsonl`，下载后再次校验。没有全量下载 SFT 文件或模型。
- 核查脚本、实样索引和机器统计保存在 `runs/data_selection_20260921/{inspect_sources.py,inspection.json,summary.json}`。使用兼容 Python 执行 `python runs/data_selection_20260921/inspect_sources.py --show` 可复查现有缓存。原始数据和这些本地产物均被 Git 忽略，跨机器需从固定来源重新获取，不是项目安装依赖。

| 新增文件 | SHA-256 |
| --- | --- |
| test/test_set.json | `9b0abc0e43805a447bc7f6e1ac8cee7a7c7fc7c89a8013df5d757a794c8b0a2b` |
| full/A Game of Thrones (A Song of Ice and Fire, #1).json | `ef1f9e69e3b6908587b98cbb9d67b731f4d2175d88eb9e54334d8f41be7d6a8d` |
| full/Pride and Prejudice.json | `ac1f8f66e85d6c90f3f1732ee633442ebe04e7711628bff2fc4e938ed16ab9d9` |

本轮支出为本地检查和公开文件下载，无付费 API、GPU 租赁、外部上传、模型训练、提交或推送。
