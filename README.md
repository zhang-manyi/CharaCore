# CharaCore

面向角色扮演模型的一致性评测与偏好优化：根据角色经历、目标、关系、相关历史和当前情境，评价并优化生成回答与行动。

计划比较基座模型、DPO 与 GRPO，使用经过校准的 LLM-as-a-Judge 提供偏好和奖励。区分稳定偏好、临时情绪、表达风格与行动，不仅靠语气或口头禅判断角色一致性。

## 文档入口

- [设计总纲](docs/DESIGN.md)：目标、范围、评价维度、实验对照和阶段交付。
- [进度与交接](docs/PROGRESS.md)：当前状态、未决事项和下一步；新对话从这里接续。
- [数据计划](docs/DATA_PLAN.md)：现成数据选择要求及当前训练输入契约。

## 当前状态

已有 TRL + PEFT LoRA/QLoRA DPO 训练器、数据契约校验和独立单元测试。当前阶段是选择现成角色情境数据，随后确定评测协议与裁判，再实施 DPO/GRPO 对照。

尚未发布正式数据集或效果结果；GRPO、裁判评分和正式统一评测尚未实现。微型训练只能验证软件机制，不代表角色能力提升。

## 本地使用

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
- scripts/：训练入口。
- tests/：使用临时合成夹具的单元测试。
- docs/：设计、数据路线与跨对话进度。
- experiments/：准备正式数据时创建版本目录；当前不附带数据集。
- data/raw/、runs/、model_cache/：本地材料、产物与权重，不纳入 Git。

代码采用 [MIT License](LICENSE)。后续接入的数据按各自来源和许可管理，代码许可不自动覆盖第三方内容。
