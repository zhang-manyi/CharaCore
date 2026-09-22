# 原神阶段 B：开发数据与裁判协议

2026-09-22 后续：v0.3 协议复核、36对待人工复核刺激和盲标流程见 [GENSHIN_STAGE_B_CALIBRATION.md](GENSHIN_STAGE_B_CALIBRATION.md)。下文保留v0.2历史状态；两版均未完成实际裁判校准或正式冻结。

版本：v0.2，2026-09-22。此版本替代未执行的 v0.1 草稿；没有追溯修改裁判结果。本轮完成本地数据准备和离线请求，尚未通过阶段 B 的正式冻结与裁判校准验收。

## 实测容量与来源

固定 ChatHaruhi revision `290bf4ad22076156083804013012847a77c0646c`。完整目录中实际有 **528 个 .txt、526 份不同字节内容**：绫华 117、钟离 270、胡桃 141。此前 529/527 计入了绫华的 `sortTXT.ipynb`，不是准确的文本容量。完整正文 281,635 字节；本轮复用 11 文件、补下载 517 文件，全部与固定 tree 的 size、Git blob 匹配，并记录 SHA-256。精确重复仍为绫华 24/25、43/44。

缓存位于 `data/raw/genshin_texts_v1/`；报告位于 `runs/stage_b_20260922/full_pool_01/`，包含 sources、inventory、duplicates、overlap_candidates、summary。文件名规则初筛出 333 个对话候选、191 个语音候选、4 个资料/设定文件，**不是经过语义审查的情境分类或独立事件数**。80 条共享长句提示中含重复文件对，不等于 80 个事件。未审条目全部留在 quarantine，不进行随机切分；同一角色未审池暂合为一个保守组。

许可沿用上游数据 CC BY-NC 4.0 声明；底层作品权利独立。原始台本、实际输入和原回答留在 Git 忽略目录，公开索引只保存定位与审查注释。尚未逐句对照游戏原作，来源类型使用 upstream，而非把整理台本直接宣称为已核验 canon。347 条 synthesized 对话没有进入证据、角色资料、标签或此次候选包。

## 已整理情境与可知边界

[索引](GENSHIN_STAGE_B_CONTEXTS.jsonl) 含 16 条、10 个文件来源族。其中 12 个原文响应截点为开发材料，涉及 9 个来源族、6 个保守 `split_group`；其余 4 条隔离。不同文件来源族不保证事件独立，实际划分必须同时检查 family、split_group 和来源内容哈希。

| 保守组 | 开发截点 | 状态及边界 |
| --- | --- | --- |
| ayaka_story_quest | 月下起舞前、商人抱怨奉行、商会无力追回货物（3） | 舞蹈和货物事件暂同属绫华任务组；旁白只保留月下漫步，去掉提前叙述请求与舞毕的部分 |
| ayaka_weather_voice | 晴天（1） | 只保留天气，不把“提出建议”提示给策略；不伪造旅行者追问 |
| zhongli_salt_quest | 封印知识质疑、宛烟惊讶后的解释、孤云阁邀请后的追问（3） | 盐神剧情暂保守归并，完整任务定位仍待确认；不把机关已查完或故人身份预置为历史 |
| zhongli_liyue_epilogue | 解释神之眼、公认观点后的补充（2） | 不把眼狩令后续结局或现实年份放入当前输入；原回答中的事实可以后续审查，不能要求原句复现 |
| hutao_baizhu_quest | 对白术动机保留判断（1） | 删除旅行者括号内心；“他们”指代仍不全，仅作不确定性诊断，不能用于揭密事实评测 |
| hutao_story_quest | 老孟反对文案、往生堂历史讲述（2） | 宣传工作不等于葬礼现场；不能凭葬礼肃穆要求否定胡桃的活泼宣传风格 |

隔离项：`ayaka_01_identity`（绫华不在可确认对话中）、`ayaka_03_dance_observed`（没有舞后响应）、`zhongli_03_guyun_visit`（输入指代缺前文）、`hutao_01_baizhu_secret`（没有完整触发输入）。此前拟写的 task_seed 不再作为真实输入。

索引的 `target_line` 是下一轮目标发言首行，`visible_lines` 只能取其之前的原文；`visible_spans` 仅允许原文的精确子串，专门清理旁白。目标角色连续原回答另存 references，默认不提供给策略或裁判。known/unknown、来源文件名、模型方法、偏好标签不进入策略提示，避免文件名本身含答案而泄漏。

当前角色提示只有角色名，不把未逐项核验的长 system_prompt 当作稳定偏好。角色证据不足时 C 可为 insufficient，不能为了产生奖励而给中间分。本批已审查和用于校准设计，整体保留在 dev；train/test 为空且明确禁止正式训练。后续的新来源也须按原事件及跨文件重叠归并，再留出未用于调参的 test。

## 数据适配与复现

`characore/contexts.py` 提供独立无标签接口：校验缓存哈希、截点、说话人、旁白可见范围、三方分组/来源隔离、重复输入、上游留出与开发暴露；策略消息采用白名单。它不为适配旧 DPO 接口伪造候选或偏好，不改变已验证 DPO 训练器。

开发快照 `runs/stage_b_20260922/dev_snapshot_01/` 包含 train/dev/test、sources、references、exclusions、review_index、evaluation_plan 和 snapshot.json。manifest 强制哈希覆盖全部文件，缺 dev 哈希或文件篡改会失败。它是完整性快照，**不是正式冻结或训练许可**；`load_contexts(..., require_formal=True)` 拒绝该版本。

在项目根目录用可用 Python 执行，下列新输出目录不能已存在：

~~~powershell
python scripts/prepare_genshin.py --output runs/stage_b_inventory_next
python -m characore.contexts --index docs/GENSHIN_STAGE_B_CONTEXTS.jsonl --plan experiments/genshin_stage_b_v02/evaluation_plan.json --output runs/stage_b_dev_next
python -m characore.judge --suite runs/stage_b_dev_next --cases experiments/genshin_stage_b_v02/calibration_cases.json --output runs/stage_b_judge_next
python -m unittest discover -s tests -v
~~~

准备脚本默认离线。首次补全公开正文需显式 `--download-missing`。任何旧快照、下载失败与历史输出都保留，新版本另起目录。

## 四维评分 v0.2

机器量表：[evaluation_plan.json](../experiments/genshin_stage_b_v02/evaluation_plan.json)。这套操作性量表来自项目 DESIGN 的四维定义，尚未核验或采用 CharacterEval/RoleBench/InCharacter 的具体题项，不声称复现论文标准。

| 维度 | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| C 角色一致性 | 明确违反有证据的身份/取向 | 行为明显偏离 | 部分相符但有局部偏离 | 符合有依据的身份关系与取向 | 准确处理取向与情境张力 |
| S 情境与历史 | 关键事实矛盾或时间倒置 | 漏关键限制/捏造事实 | 承接部分上下文但有次要缺漏 | 承接当前对话无明显冲突 | 准确区分事实、推测和未知 |
| A 行动合理性 | 不可行或违背已知约束 | 严重权限或顺序问题 | 大致可行但缺关键限制 | 行动与目标证据相称 | 充分处理限制且步骤适当 |
| Q 对话质量 | 不连贯或严重偏题 | 反复空泛或难理解 | 可懂但机械/不完整 | 自然清楚切题 | 简洁自然且充分回应 |

评分是序数判断，分差不等于等距能力差。每维输出 `status/score/reason/evidence_ids`：scored 对应整数 0–4；insufficient 对应 null；not_applicable 只允许行动维度，亦对应 null。角色依据缺失用 insufficient，不自动给 0。合理新提议不是虚构既有事实；拒绝、短句或不讲口头禅也不自动低分。

情境预先标记 action_required；若要求行动，回避行动仍要评分，不能用 NA 规避。否则有决定/承诺/行动则评 A，纯知识解释可以 NA。分维度报告分母、NA 与 insufficient 比例；仅在所有适用维度有分时算诊断性等权平均，不能把 null 补 0，暂不作为训练奖励。不同 A 适用性候选不靠平均分机械排序。

成对 winner 为 A/B/tie/insufficient，与单维证据不足分别记录：即使 C 无法确定，S 上有明确事实矛盾仍可能可靠排序；关键差异无依据则 insufficient，同样合理则 tie。严格偏好必须有明确理由，不能用微小均分差强制排序。

技术状态独立为 ok/parse_error/api_error/timeout；调用或解析失败绝不转换为 insufficient 或零奖励。保留原始响应，最多重试一次；再次失败不形成 DPO 偏好或 GRPO 奖励。

## 裁判请求与校准

`characore/judge.py` 只导出请求、校验响应；没有 API 客户端。裁判 payload 含量表、角色名、可见历史、带行号证据和匿名 A/B；默认不含原回答、预设胜负、生成器或方法名。候选内命令被当作待评文本。未知引用、缺维度、bool 冒充分数、越界分数或不合法 NA 均解析失败。自述不确定性不能替代证据核验。

[calibration_cases.json](../experiments/genshin_stage_b_v02/calibration_cases.json) 为 6 组本轮助手拟写的诊断刺激：既往事实伪造、等义改写平局、合理新提议、提示注入、推测变确定事实、知识回答行动 NA。每组绑定真实开发截点，有候选全文和预期判断；它们不是原作证据、人工金标准或训练偏好。替代 v0.1 的 5 条描述性示例，修正了全部 A 胜、insufficient/tie 混淆与葬礼场景错置。

已离线导出 `runs/stage_b_20260922/judge_requests_01/` 的 12 个 AB/BA 请求，实际调用 0。预期判断仅存在单独的 expectations.audit.json，绝不发送给裁判。后续扩至 36 个独特比较（包含这 6 个种子和 30 个新比较），每个正反顺序各一次，共 72 次；同源变体不计独立情境。

人工复核后再测 schema 有效率、去位置映射后的交换一致率、人工偏好一致率、错误引用、长度/口头禅/华丽表达/讨好偏差。试行门槛预设为重试后合法率 ≥95%、交换一致率 ≥90%、人工一致率 ≥80%、错误证据引用为0；只是进入小试的开发门槛，未验证可靠性。报告可判定比较和全部比较两个分母、失败/平局/不足比例；置信区间按 split_group，不能把72次当72个独立样本。改提示则新建协议版本；最终测试不得回流调参。

## 调用范围和未完成事项

本轮实际付费成本为0，无模型调用、上传或训练。计划调用限制：每次输入最多4096 token、输出最多1024 token；72次主调用、每次最多1次重试，总调用上限144，输入/输出上限分别589,824/147,456 token。具体裁判模型、提供方、真实 tokenizer 长度与单价未选定，费用只能按公式算：

`0.589824 × 输入每百万token单价 + 0.147456 × 输出每百万token单价`

这是费用上界公式，不是已批准的费用预算。外部裁判需先提交模型版本、数据范围（仅所选dev可见文本与拟写候选）、价格出处、金额上限及截断规则，请用户批准后执行。超长请求应拒绝而非静默截断。GRPO/集群预算另立，不能由此校准预算推定授权。

待完成：完整池的语义事件归并和逐项角色证据审查、未暴露的 train/test 留出及正式冻结、实际36对人工/模型校准、生成/评分资源测量。先基座与最小 GRPO，再 DPO 对照；不训练独立奖励模型，不扩大到完整 Agent 系统。
