# Qwen3-8B、双 V100 与 API 裁判

策略模型固定为 ModelScope `Qwen/Qwen3-8B`，裁判配置为 OpenAI API 的 `gpt-6-astra`。`.env.example` 使用官方地址 `https://api.openai.com/v1` 和 Chat Completions。账户模型权限与实际调用仍需验证；配置检查不会联网，必须显式 `--allow-api` 才发送裁判材料。

## 1. 更新代码、下载策略模型

在分配给你的集群节点、已有 `characore` 环境中：

```bash
cd ~/projects/CharaCore
conda activate characore
git pull --ff-only origin main
python -m unittest discover -s tests -v
python -m pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir "$HOME/models/Qwen3-8B"
export CHARACORE_POLICY_MODEL="$HOME/models/Qwen3-8B"
```

已有完整下载时直接设置路径，无需重复下载。不需要安装 vLLM，不需要下载本地裁判模型；API 适配器使用 Python 标准库。推理和训练加载本地模型时继续禁止自动联网下载。

## 2. 填写 API 密钥

```bash
test -f .env || cp .env.example .env
chmod 600 .env
nano .env
```

只修改你实际使用的参数，特别是密钥：

```dotenv
CHARACORE_JUDGE_BASE_URL=https://api.openai.com/v1
CHARACORE_JUDGE_API_KEY=这里填你的OpenAI_API密钥
CHARACORE_JUDGE_MODEL=gpt-6-astra
CHARACORE_JUDGE_API_STYLE=chat_completions
CHARACORE_JUDGE_MAX_TOKENS=4096
CHARACORE_JUDGE_TIMEOUT_SECONDS=120
CHARACORE_JUDGE_MAX_CALLS=100
CHARACORE_JUDGE_MAX_INPUT_BYTES=16384
CHARACORE_JUDGE_JSON_MODE=true
```

程序自动读取项目根目录 `.env`，无需 `source .env`；已导出的同名环境变量优先。`.env` 被 Git 忽略，`.env.example` 纳入版本控制。不要把密钥写进 Python、命令行参数或文档。若选择 Responses，设置 `API_STYLE=responses`；不要把完整 `/chat/completions` 或 `/responses` 路径填入BASE_URL。

## 3. 先检查配置，再调用一次

```bash
# 不联网、不扣费，不显示密钥
python scripts/check_judge_api.py

# 会向配置的API发送一个原创合成案例；最多一次请求，不自动重试
python scripts/check_judge_api.py --allow-api --output runs/api_smoke_01
```

第二条正常时输出 `status=ok`；这只验证连通性和JSON格式，`calibration_passed`仍为false（裁判可靠性未校准，见第8节）。`runs/api_smoke_01/call/`保存请求、原始模型返回、用量及严格解析结果。404需检查API路径/模型ID，401/403检查密钥及账户权限，400检查接口支持的参数；不得自动换模型或补造必需字段。JSON mode关闭时仍执行本地严格解析。需要进一步诊断时提供脱敏失败记录，不发送`.env`。

官方[GPT-6 Astra模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)列出`gpt-6-astra`；API样式对应官方 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 与 [Responses](https://developers.openai.com/api/reference/resources/responses/methods/create) 的请求格式。不强制temperature或reasoning参数；记录实际返回的model和usage。4096输出预算可能包含推理token，若返回不完整会拒绝评分，不擅自加预算。请求设置`store=false`。

## 4. 跑 Qwen3-8B 真实基线

```bash
CUDA_VISIBLE_DEVICES=0 python -B scripts/run_agent.py run \
  --policy local --base "$CHARACORE_POLICY_MODEL" \
  --device cuda --precision fp16 \
  --max-steps 12 --max-new-tokens 192 \
  --output runs/qwen3_8b_baseline_01
```

这一步只推理，不训练，不调用裁判API。V100 compute capability 7.0不原生支持BF16；`--precision auto`也会选FP16，而不依据可能包含模拟支持的`is_bf16_supported()`。返回码2表示模型任务未完成，先看轨迹，不等同于程序安装失败。

```bash
python scripts/run_agent.py replay runs/qwen3_8b_baseline_01/episode/trajectory.json
```

## 5. 验证双进程，再准备真实双卡训练

Linux节点可先运行无模型、无API的CPU双进程检查：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py \
  --tiny --steps 2 --output runs/cluster_ddp_tiny_01
```

检查两个 `rank_0000/verification.json`、`rank_0001/verification.json` 的 `world_size=2`、`optimizer_steps=2`、`ddp_adapter_parameters_equal=true` 及原来的五项更新证明。这个检查证明进程同步和CPU训练，不证明V100显存或吞吐。本机已用Gloo/FileStore验证两进程更新；Windows torchrun/TCPStore因libuv构建限制未运行成功，不建议把该Windows启动方式照搬到Linux。

## 6. 用桩裁判跑通完整训练路径

训练数据随仓库提供在 `experiments/agent_v1/suite`（71 训练 / 4 评测检查点，由 `scripts/build_agent_suite.py` 在真实环境上 BFS 枚举生成，`git pull` 即可获得，无需另行准备）。先用不联网的桩裁判确认整条链路：

```bash
python scripts/train_grpo.py --preflight-only \
  --base "$CHARACORE_POLICY_MODEL" --suite experiments/agent_v1/suite \
  --judge-backend stub --output runs/preflight_01

CUDA_VISIBLE_DEVICES=0 python -B scripts/train_grpo.py \
  --base "$CHARACORE_POLICY_MODEL" --device cuda --precision fp16 \
  --judge-backend stub --suite experiments/agent_v1/suite \
  --group-size 4 --steps 2 --output runs/qwen3_8b_grpo_stub_01
```

`--preflight-only` 不请求API、不加载模型、不更新参数。桩裁判只比较环境执行是否成功，用来验证请求构造、严格解析、AB/BA 一致性和奖励装配；**它不是质量信号**，这一步的产物不能当作角色一致性结果。这一步通过后，剩下的未知量只有显存和真实API。

## 7. 真实API裁判的双卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py \
  --base "$CHARACORE_POLICY_MODEL" --device cuda --precision fp16 \
  --judge-backend api --allow-api \
  --suite experiments/agent_v1/suite \
  --group-size 4 --steps 2 --output runs/qwen3_8b_grpo_01
```

策略采用FP16基座+FP32 LoRA训练参数，TRL处理混合精度；每卡microbatch=1、累积4步、每组4样本。每进程生成一组，同步全局奖励和梯度；双卡各持有完整8B基座，不合并成64GB显存。单卡32GB的实际峰值仍需测量；显存不足时先降 `--group-size` 到 2。

按rank独立保存请求/返回/适配器，避免双进程覆盖。每个进程100次API尝试上限包含重试；每个样本需要AB/BA两次调用，因此每进程每步约 `group_size × 2` 次，请据此核对 `CHARACORE_JUDGE_MAX_CALLS`。策略复现参考动作时判为平局且不发起调用。奖励异常先跨进程同步再同时拒绝更新，避免一边反向传播、一边已退出。失败仍留档；同分组不虚造学习信号。

## 8. 产物与限度

`runs/<name>/verification.json` 给出优化步数、LoRA参数变化、梯度有限非零、冻结参考未变、适配器重载一致以及DDP跨rank一致这几项证明；`before_development_eval.json` 与 `after_development_eval.json` 给出训练前后在4个评测检查点上的动作与执行结果。

限度需要一并说明：优化单位是**单步动作**，不是完整episode；评测检查点与训练检查点来自同一个原创任务族，是开发集而非独立盲测；API裁判与人工判断的一致率尚未测量，因此奖励可靠性未经校准。`runs/` 和模型不随Git下载。
