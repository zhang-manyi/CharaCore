"""TRL 0.26.2 GRPO/PEFT entry: two-character speech style vs frozen base replies, or CPU random-model mechanics.

Single process, single GPU. Rejected groups (judge failure, parse error, AB/BA
disagreement) are masked out of the update by StyleGRPOTrainer and counted;
they are never zero-filled.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from characore.persona import render_prompt
from characore.protocol import digest, dump
from characore.precision import select_precision
from characore.style_data import load_base_replies, load_suite
from characore.style_rewards import REWARD_SPEC

SOURCES = ("scripts/train_grpo.py", "characore/style_rewards.py", "characore/style_trainer.py",
           "characore/style_data.py", "characore/persona.py", "characore/judge.py")


class TinyReward:
    """Arbitrary token-ID mean: tests sampled rewards, gradients and group masking, NOT task quality.

    Rejects group 0 of the first batch so the masking path runs on every tiny check.
    """
    __name__ = "tiny"

    def __init__(self, vocab, output):
        self.vocab, self.output, self.batches = vocab, output, 0
        self.pending_rejected, self.last_metrics = None, {}

    def take_rejected(self):
        rejected, self.pending_rejected = self.pending_rejected, None
        return rejected

    def __call__(self, prompts, completions, completion_ids, **kwargs):
        self.batches += 1
        values = [sum(ids) / max(len(ids), 1) / self.vocab for ids in completion_ids]
        self.pending_rejected = [0] if self.batches == 1 else []
        dump(self.output / f"sampled_rewards_{self.batches:03d}.json",
             dict(completions=completions, token_ids=completion_ids, values=values, rejected=self.pending_rejected))
        return values


def preflight(args):
    """Validate the frozen suite and base replies. Judge reliability is reported, never assumed."""
    if args.tiny:
        return None
    train, test = load_suite(args.suite)
    return train, test, load_base_replies(args.base_replies, args.suite)


def run(args, checked):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import set_seed
    from trl import GRPOConfig
    from characore.model_loader import load_model
    from characore.style_trainer import make_trainer_class

    if importlib.metadata.version("trl") != "0.26.2":
        raise ValueError("this adapter is verified against TRL 0.26.2")
    torch.set_num_threads(4)
    set_seed(args.seed)
    load_device = "cuda:0" if args.device == "cuda" else "cpu"
    precision = select_precision(load_device, args.precision)
    model, tokenizer = load_model(args.base, tiny=args.tiny, device=load_device, precision=precision)
    model.config.use_cache = True
    judge = None
    if args.tiny:
        model.save_pretrained(args.output / "tiny_base")
        tokenizer.save_pretrained(args.output / "tiny_base")
        rows = [{"prompt": "medic help patient "}, {"prompt": "guard bridge safe "}]
        reward = TinyReward(len(tokenizer), args.output)
        targets, completion_length = ["c_attn", "c_proj"], 8
    else:
        from characore.style_rewards import StyleReward
        if args.judge_backend == "api":
            from characore.api_judge import APIJudge
            judge = APIJudge(args.env_file, allow_calls=args.allow_api)
            needed = args.steps * args.prompts_per_step * args.group_size * 2
            if judge.max_calls < needed:
                print(json.dumps(dict(warning="CHARACORE_JUDGE_MAX_CALLS below worst case",
                                      max_calls=judge.max_calls, worst_case=needed)), flush=True)
        else:
            from characore.stub_judge import StubJudge
            judge = StubJudge()
        train, _, base_replies = checked
        completion_length = args.max_new_tokens
        limit = model.config.max_position_embeddings
        rows = [dict(prompt=render_prompt(tokenizer, r, completion_length, limit), row_id=r["id"],
                     character=r["character"], base_reply=base_replies[r["id"]], situation=r["situation"],
                     speaker=r["speaker"], line=r["line"]) for r in train]
        reward = StyleReward(judge, args.output / "reward", args.group_size, workers=args.judge_workers)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    generation_batch = args.prompts_per_step * args.group_size
    if generation_batch % args.micro_batch:
        raise ValueError("prompts_per_step * group_size must be divisible by micro_batch")
    # One generation batch per optimizer step: steps_per_generation == gradient_accumulation_steps.
    cfg = GRPOConfig(output_dir=str(args.output / "trainer"), max_steps=args.steps,
                     per_device_train_batch_size=args.micro_batch,
                     gradient_accumulation_steps=generation_batch // args.micro_batch,
                     generation_batch_size=generation_batch,
                     num_generations=args.group_size, max_completion_length=completion_length,
                     learning_rate=5e-4 if args.tiny else args.learning_rate, beta=.04, loss_type="grpo",
                     scale_rewards="group", temperature=1.0, top_p=1.0, top_k=0,
                     logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
                     seed=args.seed, data_seed=args.seed, use_cpu=args.device == "cpu",
                     bf16=precision == "bf16", fp16=precision == "fp16", gradient_checkpointing=not args.tiny,
                     gradient_checkpointing_kwargs={"use_reentrant": False},
                     dataloader_pin_memory=False, remove_unused_columns=False, disable_tqdm=True,
                     mask_truncated_completions=False, use_vllm=False)
    trainer = make_trainer_class()(model=model, reward_funcs=reward, args=cfg, style_reward=reward,
                                   train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                                   peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                                          lora_dropout=0.0, target_modules=targets,
                                                          task_type="CAUSAL_LM"))
    policy = trainer.model
    trainables = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    if not trainables or any("lora_" not in n for n in trainables):
        raise AssertionError("expected LoRA-only updates")
    before = {n: p.detach().cpu().clone() for n, p in trainables.items()}
    grads = dict(calls=0, nonzero=0, finite=True)

    def observe(g):
        grads["calls"] += 1
        grads["nonzero"] += bool(torch.count_nonzero(g))
        grads["finite"] &= bool(torch.isfinite(g).all())

    hooks = [p.register_hook(observe) for p in trainables.values()]
    probe = tokenizer(rows[0]["prompt"], return_tensors="pt").to(policy.device)
    probe = {k: v[:, -8:] for k, v in probe.items() if k in ("input_ids", "attention_mask")}

    def logits():
        policy.eval()
        amp = torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16) if precision != "fp32" else nullcontext()
        with torch.no_grad(), amp:
            return policy(**probe).logits.detach().float().cpu()

    initial = logits()
    with policy.disable_adapter():
        ref_before = logits()
    dump(args.output / "manifest.json", dict(
        framework="TRL GRPOTrainer + PEFT, StyleGRPOTrainer group rejection", trl="0.26.2",
        source_sha256={p: digest(ROOT / p) for p in SOURCES},
        tiny=args.tiny, claim="software mechanics only" if args.tiny else "speech-style GRPO vs frozen base replies",
        device=load_device, precision=precision,
        suite_sha256=None if args.tiny else digest(args.suite / "freeze.json"),
        base_replies_sha256=None if args.tiny else digest(args.base_replies / "freeze.json"),
        judge_backend=None if args.tiny else args.judge_backend,
        judge_reliability="uncalibrated: no human agreement measured for this task",
        judge_metadata=judge.metadata if judge else None,
        base_files_sha256={p.name: digest(p) for p in sorted((args.output / "tiny_base" if args.tiny else Path(args.base)).iterdir()) if p.is_file()},
        seed=args.seed, group_size=args.group_size, prompts_per_step=args.prompts_per_step, steps=args.steps,
        config=cfg.to_dict(), reward_spec=REWARD_SPEC if not args.tiny else "synthetic token-ID mean",
        packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate")}))
    try:
        result = trainer.train()
    finally:
        if not args.tiny:
            # Curves and counts survive an aborted run too.
            dump(args.output / "reward_totals.json", dict(reward.totals, rejected_by_trainer=trainer.rejected_total))
        dump(args.output / "log_history.json", trainer.state.log_history)
    for hook in hooks:
        hook.remove()
    trained = logits()
    with policy.disable_adapter():
        ref_after = logits()
    changed = sum(not torch.equal(before[n], p.detach().cpu()) for n, p in trainables.items())
    if not changed or not grads["finite"] or not grads["nonzero"] or torch.equal(initial, trained):
        raise AssertionError("no finite nonzero gradient/update evidence")
    if not torch.equal(ref_before, ref_after):
        raise AssertionError("frozen reference changed")
    policy.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    policy = PeftModel.from_pretrained(policy.unload(), args.output / "adapter", is_trainable=False)
    restored = logits()
    if not torch.allclose(trained, restored, atol=1e-5, rtol=1e-5):
        raise AssertionError("adapter save/reload mismatch")
    proof = dict(software_mechanism_only=args.tiny, optimizer_steps=trainer.state.global_step,
                 ddp_adapter_parameters_equal="not_applicable_single_process",
                 changed_tensors=changed, gradients=grads, reference_logits_unchanged=True,
                 probe_logit_max_delta=(trained - initial).abs().max().item(),
                 reload_logit_max_error=(restored - trained).abs().max().item(),
                 rejected_groups=trainer.rejected_total,
                 reward_totals=None if args.tiny else reward.totals,
                 adapter_sha256=digest(args.output / "adapter/adapter_model.safetensors"),
                 metrics=result.metrics)
    dump(args.output / "verification.json", proof)
    print(json.dumps(proof, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--base-replies", type=Path)
    parser.add_argument("--judge-backend", choices=("api", "stub"), default="api")
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.group_size < 2:
        parser.error("GRPO needs at least two samples per group")
    if args.tiny:
        if args.device != "cpu" or args.allow_api or any((args.base, args.suite, args.base_replies)):
            parser.error("tiny mode is CPU-only random model; cannot mix real inputs")
        args.group_size, args.prompts_per_step, args.micro_batch = 2, 2, 1
    elif not all((args.base, args.suite, args.base_replies)):
        parser.error("real mode requires --base, --suite and --base-replies")
    if not args.tiny and args.judge_backend == "api" and not args.allow_api and not args.preflight_only:
        parser.error("API reward training requires --allow-api")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("single process only: group rejection is not verified under DDP")
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "invocation.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        checked = preflight(args)
        if args.preflight_only:
            dump(args.output / "preflight.json", dict(passed=True, training_started=False, tiny=args.tiny,
                                                      train_rows=None if args.tiny else len(checked[0])))
        else:
            run(args, checked)
    except Exception:
        dump(args.output / "failure.json", dict(traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
