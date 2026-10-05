# 集群快速开始：Qwen3-4B、单张 V100、API 裁判

集群节点不能访问 GitHub，代码用 git bundle 传输。以下命令都是单行，可逐条粘贴；长任务放进 tmux。

## 1. 传代码

开发机（仓库根目录）：

```bash
git bundle create ../characore.bundle main
scp -P 20819 ../characore.bundle zhangmy@172.20.10.250:/home/zhangmy/projects/
```

集群：

```bash
cd /home/zhangmy/projects/CharaCore
conda activate characore
git pull ../characore.bundle main
python -m unittest discover -s tests
```

## 2. 下载策略模型（只需一次）

```bash
modelscope download --model Qwen/Qwen3-4B --local_dir /home/zhangmy/models/Qwen3-4B
export CHARACORE_POLICY_MODEL=/home/zhangmy/models/Qwen3-4B
```

加载一律离线，不会自动下载；路径不对会直接报错。

## 3. 配置裁判

`.env` 不随 bundle 传输，在集群上单独维护（`cp .env.example .env`，`chmod 600 .env`）。中转地址与模型：

```dotenv
CHARACORE_JUDGE_BASE_URL=https://naiccc.com/v1
CHARACORE_JUDGE_MODEL=gpt-6-astra
CHARACORE_JUDGE_API_STYLE=chat_completions
CHARACORE_JUDGE_MAX_CALLS=6000
```

`MAX_CALLS` 按进程计；每步最多 `prompts_per_step × group_size × 2` 次比较（默认 2×8×2=32，150 步约 4800 次），同批复用、硬违规与逐字节相同不发起调用，实际更少。调用或解析失败重试 1 次，瞬时网络故障另有退避重试（见第 7 节），重试都计入预算；按 survey_style_02 约 3% 的失败率，额外开销在百次以内。续训时预算重新计数。

```bash
python scripts/check_judge_api.py
python scripts/check_judge_api.py --allow-api --output runs/api_smoke_01
```

第一条不联网；第二条只发一个合成请求，验证连通与严格解析。

## 4. 冻结基座回复（只需一次）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_base_replies.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --output experiments/style_v1_base_qwen3_4b
```

产物 `replies.json` 与 `freeze.json` 应提交回仓库（集群上 `git add` 后用 bundle 带回，或 scp 回开发机），之后所有训练与评测都绑定它。

## 5. 裁判顺序一致性与重复一致率

```bash
python scripts/check_judge_order.py --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --output runs/judge_order_01 --cases 10 --allow-api
```

30 次调用。看 `summary.json` 的 `usable_rate`（AB/BA 一致）与 `repeat_agreement`（相同请求两次结论相同）。

## 6. survey

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/survey_sampling.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --output runs/survey_01 --judge-backend api --allow-api --judge-rows 20
```

最多 320 次比较（失败重试另计）。`summary.json` 的 `gate.passed` 为 true 再训练；否则看 `varied_share`、`rejected_share`、`win_rate_vs_base`、`judged.order_inconsistent_rate`、`hard_rate` 再调整。

## 7. 桩裁判跑通训练路径，再真实训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --judge-backend stub --device cuda --precision fp16 --steps 3 --output runs/grpo_stub_01
```

桩裁判按规则分判定，只验证链路与显存，不是质量信号。通过后在 tmux 里跑真实训练：

```bash
tmux new -s grpo
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --judge-backend api --allow-api --device cuda --precision fp16 --steps 150 --output runs/grpo_style_01
```

`Ctrl-b d` 脱离，`tmux attach -t grpo` 回来。显存不足时加 `--micro-batch 2`。

裁判 API 的超时、断连、429、5xx 会自动退避重试约 8 分钟。仍然中断（额度耗尽、断电、Ctrl-C）时，用同一条命令加 `--resume` 从最近的检查点（每 10 步一个，`trainer/checkpoint-*`）继续，其余参数必须不变：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --judge-backend api --allow-api --device cuda --precision fp16 --steps 150 --output runs/grpo_style_01 --resume
```

续训的日志、`log_history.json`、`reward_totals.json` 在 `runs/grpo_style_01/resume_01/`；`log_history.json` 覆盖从第 1 步起的全部步数。

双卡：把 `CUDA_VISIBLE_DEVICES=0 python` 换成 `CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16`，其余参数不变（每步仍是 2 × 8 个样本，每卡 8 个）：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 scripts/train_grpo.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --judge-backend stub --device cuda --precision fp16 --steps 3 --output runs/grpo_stub_ddp_01
```

逐批记录分在 `reward/rank0/`、`reward/rank1/`；rank 1 的 `reward_totals_rank1.json`、`failure_rank1.json` 与 rank 0 的文件放在一起。续训同样加 `--resume`，卡数必须与原运行相同。

## 8. 评测

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_style.py --base $CHARACORE_POLICY_MODEL --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --output runs/eval_base_01 --allow-api
CUDA_VISIBLE_DEVICES=0 python scripts/eval_style.py --base $CHARACORE_POLICY_MODEL --adapter runs/grpo_style_01/adapter --suite experiments/style_v1 --base-replies experiments/style_v1_base_qwen3_4b --output runs/eval_grpo_01 --allow-api
```

每次约 360 次调用。`report.json` 给胜率、规则指标、串角色率；`side_by_side.md` 是前后对照。训练曲线在 `runs/grpo_style_01/log_history.json` 的 `style/*` 字段，逐批明细在 `reward/rewards_*.json`。

## 产物与限度

`verification.json`：优化步数、LoRA 参数变化、梯度有限非零、冻结参考不变、适配器重载一致；双卡时 `ddp_adapter_parameters_equal` 为实测的两卡 LoRA 一致性，单卡为不适用。裁判与人工一致率未测量，裁判类指标都是 uncalibrated。`runs/` 与模型不进 Git。
