# CharaCore

面向角色行动一致性的强化学习优化。在单角色短任务中接入工具调用、显式状态和事件记忆，用 LLM-as-a-Judge 的成对偏好作为奖励，通过 GRPO 优化策略在具体状态下的下一步动作。

主线是**奖励与强化学习机制**：环境给出可验证的执行结果，裁判给出成对偏好，两者组合成奖励；奖励一旦不可用就中止更新，不用零值顶替。角色一致性区分稳定偏好、临时情绪、表达风格与实际行动，不只看语气或口头禅。

## 文档入口

- [设计总纲](docs/DESIGN.md)：目标、范围、数据接口、评价维度、实验对照和阶段交付。
- [集群快速开始](docs/CLUSTER_QUICKSTART.md)：Qwen3-8B + 双 V100 + API 裁判的完整训练路径。
- [GRPO 接入说明](docs/GRPO_LOCAL.md)：奖励定义、数据契约与失败处理。
- [进度与交接](docs/PROGRESS.md)：当前状态、未决事项和下一步。

## 当前状态

Agent 环境、奖励、训练入口和训练数据都已就绪：`experiments/agent_v1/suite` 随仓库提供 71 个训练检查点和 4 个评测检查点，由 `scripts/build_agent_suite.py` 在真实环境上 BFS 枚举生成，克隆后即可直接训练。

已实现单角色短任务、三类本地工具、显式状态、事件记忆和 JSON 轨迹回放。原创角色“岚”的密封档案递送任务是设计测试环境。CPU 随机微型模型的 2 步更新已验证软件机制；真实 Qwen3-8B 的 GRPO 训练在 GPU 集群执行。

限度：优化单位是**单步动作**，不是完整 episode；评测检查点与训练检查点同属一个原创任务族，是开发集而非独立盲测；API 裁判与人工判断的一致率尚未测量，奖励可靠性未经校准。微型训练只验证软件机制，不代表角色能力提升。

## 本地使用

最小离线闭环只需 Python 标准库，不必安装训练依赖。在根目录执行（输出目录必须尚不存在）：

~~~powershell
python scripts/run_agent.py verify --output runs/agent_verify_01
python scripts/run_agent.py run --policy scripted --scenario commitment --output runs/agent_demo_01
python scripts/run_agent.py replay runs/agent_demo_01/episode/trajectory.json
python -m unittest discover -s tests -v
~~~

`verify` 运行正常完成、非法调用拒绝、跨轮承诺三条脚本轨迹，并重新执行校验回放；这些只验证软件机制。`run` 每轮保存可见输入、原始策略输出、工具参数/返回、状态变化及失败。默认最多12轮，解析错误也消耗步数；完成、放弃、推理异常或步数耗尽后终止。回放不加载模型，会检查输入、状态、返回和统计是否一致。

重新生成训练套件（可选，仓库已包含产物）：

~~~powershell
python scripts/build_agent_suite.py --output runs/agent_suite_01
~~~

真实推理复用已有 Transformers 环境及本地模型目录，严格离线，不自动下载：

~~~powershell
# python须指向已安装requirements.txt中推理依赖的环境；模型路径是本地占位示例
python -B scripts/run_agent.py run --policy local --base model_cache/Qwen3-1.7B --device cuda --max-steps 12 --max-new-tokens 192 --output runs/agent_base_01
~~~

可选 `--quantize` 使用现有4-bit加载路径。返回码0表示任务/验证完成，2表示任务未完成，1表示初始化或运行异常。已存在输出目录会拒绝复用。`trajectory.json` 是完整回放入口，`metadata.json` 记录模型/代码哈希与推理参数；初始化失败保存在 `failure.json`。规则统计区分模型违规尝试与环境拒绝后的安全状态，不能把工具调用成功率当任务完成率，也不能把封条被环境保护当作模型主动守诺。

## GRPO 训练

集群使用 ModelScope Qwen3-8B、双 V100 和 API 裁判，按[集群快速开始](docs/CLUSTER_QUICKSTART.md)操作。复制 `.env.example` 到被忽略的 `.env` 填写 API 密钥；API 默认不调用，须显式 `--allow-api`。

准备 Python 3.12 训练环境：

~~~powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

requirements.txt 固定的训练依赖面向已验证的 Windows/CUDA 12.1 配置，不是跨平台完整锁文件。其他平台需选择匹配的 PyTorch 构建。

随机微型模型的机制检查（CPU，不需要模型或 API）：

~~~powershell
.venv\Scripts\python.exe scripts/train_grpo.py --tiny --steps 2 --output runs/tiny_grpo_01
~~~

真实模型训练，先用不联网的桩裁判跑通链路，再换真实裁判：

~~~powershell
.venv\Scripts\python.exe scripts/train_grpo.py --base model_cache/Qwen3-8B --suite experiments/agent_v1/suite --judge-backend stub --device cuda --precision fp16 --steps 2 --output runs/grpo_stub_01
~~~

桩裁判只比较环境执行是否成功，用于验证请求构造、严格解析、AB/BA 顺序一致性和奖励装配，**不是质量信号**。每次使用新的输出目录。训练器记录数据与模型哈希、环境版本、梯度和参数更新，并检查冻结参考、DDP 跨 rank 一致性及适配器保存重载。

## 目录

- characore/：环境、奖励、裁判协议与数据契约。
- scripts/：训练、套件生成、Agent 运行与 API 检查入口。
- tests/：使用临时合成夹具的单元测试。
- docs/：设计、集群操作与跨对话进度。
- experiments/：版本化评测计划与冻结训练套件。
- data/raw/、runs/、model_cache/：本地材料、产物与权重，不纳入 Git。

原神来源审核、Stage B 校准材料与 DPO 训练器保留在 `genshin-data-audit` 分支，不在主线。

代码采用 [MIT License](LICENSE)。后续接入的数据按各自来源和许可管理，代码许可不自动覆盖第三方内容。
