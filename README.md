# CharaCore

面向角色回答与行动一致性的评测与优化。近期目标是在单角色短任务中接入工具、显式状态和事件记忆，观察模型的决策与实际执行结果。

计划比较基座模型、DPO 与 GRPO，使用经过校准的 LLM-as-a-Judge 提供偏好和奖励。区分稳定偏好、临时情绪、表达风格与行动，不仅靠语气或口头禅判断角色一致性。

## 文档入口

- [设计总纲](docs/DESIGN.md)：目标、范围、数据接口、评价维度、实验对照和阶段交付。
- [进度与交接](docs/PROGRESS.md)：当前状态、未决事项和下一步；新对话从这里接续。

## 当前状态

已有 TRL + PEFT LoRA/QLoRA DPO 训练器、数据契约、原神来源审核、无标签开发快照及[36对离线校准材料](docs/GENSHIN_STAGE_B_CALIBRATION.md)。正式 train/test 仍为空，人工与模型校准均未执行，裁判模型未选定。

已实现单角色短任务、三类本地工具、显式状态、事件记忆和 JSON 轨迹回放。原创角色“岚”的密封档案递送任务明确属于设计测试环境，不是原神原作剧情。脚本验证通过；本地 Qwen3-1.7B 已真实运行一次，在重复承诺中耗尽12步，未完成递送。完整结果见[进度与交接](docs/PROGRESS.md)，不据单次开发运行声称能力改善。真实裁判奖励训练仍需数据与校准验收。

尚未发布正式数据集或效果结果。已复用 TRL/PEFT 接入单步动作 GRPO、裁判执行/校准统计与奖励失败处理，并通过 CPU 随机微型模型的2步更新验证；真实任务训练未启动。现有本地模型的2条裁判试跑均未通过严格解析，人工校准仍未执行。操作与限制见 [GRPO本地接入](docs/GRPO_LOCAL.md)。微型训练只能验证软件机制，不代表角色能力提升。

## 本地使用

最小离线闭环只需 Python 标准库，不必安装训练依赖。在根目录执行（输出目录必须尚不存在）：

~~~powershell
python scripts/run_agent.py verify --output runs/agent_verify_01
python scripts/run_agent.py run --policy scripted --scenario commitment --output runs/agent_demo_01
python scripts/run_agent.py replay runs/agent_demo_01/episode/trajectory.json
python -m unittest discover -s tests -v
~~~

`verify` 运行正常完成、非法调用拒绝、跨轮承诺三条脚本轨迹，并重新执行校验回放；这些只验证软件机制。`run` 每轮保存可见输入、原始策略输出、工具参数/返回、状态变化及失败。默认最多12轮，解析错误也消耗步数；完成、放弃、推理异常或步数耗尽后终止。回放不加载模型，会检查输入、状态、返回和统计是否一致。

真实推理复用已有 Transformers 环境及本地模型目录，严格离线，不自动下载：

~~~powershell
# python须指向已安装requirements.txt中推理依赖的环境；模型路径是本地占位示例
python -B scripts/run_agent.py run --policy local --base model_cache/Qwen3-1.7B --device cuda --max-steps 12 --max-new-tokens 192 --output runs/agent_base_01
~~~

可选 `--quantize` 使用现有4-bit加载路径；本次实际运行使用 BF16、未量化。返回码0表示任务/验证完成，2表示任务未完成，1表示初始化或运行异常。已存在输出目录会拒绝复用。`trajectory.json` 是完整回放入口，`metadata.json` 记录模型/代码哈希与推理参数；初始化失败保存在 `failure.json`。规则统计区分模型违规尝试与环境拒绝后的安全状态，不能把工具调用成功率当任务完成率，也不能把封条被环境保护当作模型主动守诺。

以下是已有 DPO 路径，仅在另行开展训练时使用；运行上述闭环不启动训练。

在项目根目录准备 Python 3.12 环境：

~~~powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

requirements.txt 固定的训练依赖面向已验证的 Windows/CUDA 12.1 配置，不是跨平台完整锁文件。其他平台需选择匹配的 PyTorch 构建。也可使用已有兼容环境；无需复制模型缓存。

可选的随机微型模型机制检查：

~~~powershell
.venv\Scripts\python.exe scripts/train_dpo.py --tiny --steps 4 --output runs/tiny_check_01
~~~

准备好本地模型与符合数据契约的目录后，使用真实模型训练。下面路径是占位示例，项目不会自动下载模型或生成正式数据：

~~~powershell
.venv\Scripts\python.exe scripts/train_dpo.py --base model_cache/Qwen3-1.7B --suite experiments/role_v1 --device cuda --quantize --steps 4 --output runs/dpo_01
~~~

每次使用新的输出目录。训练器记录数据与模型哈希、环境版本、梯度和参数更新，并检查冻结参考及适配器保存重载。四步命令是机制试运行示例，正式训练预算在数据与评测确定后设置。

## 目录

- characore/：数据契约、提示构造与偏好转换。
- scripts/：训练、文本准备与离线来源审核入口。
- tests/：使用临时合成夹具的单元测试。
- docs/：设计、数据路线与跨对话进度。
- experiments/：版本化裁判计划与开发刺激；当前不附带正式训练数据集。
- data/raw/、runs/、model_cache/：本地材料、产物与权重，不纳入 Git。

代码采用 [MIT License](LICENSE)。后续接入的数据按各自来源和许可管理，代码许可不自动覆盖第三方内容。
