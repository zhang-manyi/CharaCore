"""TRL 0.26.2 GRPO/PEFT entry: explicit next-action data, or CPU random-model mechanics."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from characore.agent import dump, messages
from characore.grpo_data import load_training_suite, verify_calibration
from characore.grpo_rewards import ActionReward, REWARD_SPEC, restore, transition, validate_groups
from characore.judge_runner import identity, judge_identity
from characore.protocol import digest


def preflight(args):
    if args.tiny:
        return None
    rows, evaluation, approval = load_training_suite(args.suite)
    report = verify_calibration(args.packet, args.judge_results, args.human)
    if approval.get("calibration_report_sha256") != identity(report):
        raise ValueError("reward approval not bound to actual calibration results")
    return rows, evaluation, report


def run(args, checked):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    from scripts.train_dpo import load_model

    if importlib.metadata.version("trl") != "0.26.2":
        raise ValueError("this adapter is verified against TRL 0.26.2")
    torch.set_num_threads(4)
    set_seed(args.seed)
    group_size = 4 if args.tiny else 2
    judge_policy = None
    if not args.tiny:
        from characore.local_policy import LocalPolicy
        judge_policy = LocalPolicy(args.judge_base, device="cpu", max_new_tokens=1024)
        if judge_identity(judge_policy.metadata) != checked[2]["judge_identity"]:
            raise ValueError("judge model/decoding/device differs from calibrated configuration")
    model, tokenizer = load_model(args.base, tiny=args.tiny, device=args.device, quantize=args.quantize)
    model.config.use_cache = True
    if args.tiny:
        model.save_pretrained(args.output / "tiny_base")
        tokenizer.save_pretrained(args.output / "tiny_base")
        rows = [{"prompt": "medic help patient "}, {"prompt": "guard bridge safe "}]
        reward_batches = []

        def reward(completions, completion_ids, **kwargs):
            # Arbitrary token-ID mean tests sampled rewards and gradients, NOT task quality.
            values = [sum(ids) / max(len(ids), 1) / len(tokenizer) for ids in completion_ids]
            reward_batches.append(dict(completions=completions, token_ids=completion_ids, values=values))
            dump(args.output / f"sampled_rewards_{len(reward_batches):03d}.json", reward_batches[-1])
            return validate_groups(values, group_size)

        targets = ["c_attn", "c_proj"]
        completion_length = 8
    else:
        rows = []
        for row in checked[0]:
            prompt = tokenizer.apply_chat_template(messages(restore(row["prefix"]).observation()),
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + 192 > model.config.max_position_embeddings:
                raise ValueError("prompt too long; refusing silent truncation")
            rows.append(dict(prompt=prompt, prefix=row["prefix"], anchor=row["anchor"]))
        rubric = json.loads((ROOT / "experiments/genshin_stage_b_v03/evaluation_plan.json").read_text(encoding="utf8"))["rubric"]
        reward = ActionReward(judge_policy, args.output / "reward_calls", rubric, group_size)
        targets, completion_length = ["q_proj", "v_proj"], 192
    cfg = GRPOConfig(output_dir=str(args.output / "trainer"), max_steps=args.steps,
                     per_device_train_batch_size=group_size, gradient_accumulation_steps=1,
                     num_generations=group_size, max_completion_length=completion_length,
                     learning_rate=5e-4 if args.tiny else 5e-5, beta=.04, loss_type="grpo",
                     scale_rewards="group", temperature=1.0, top_p=1.0, top_k=0,
                     logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
                     seed=args.seed, data_seed=args.seed, use_cpu=args.device == "cpu",
                     bf16=args.device == "cuda", fp16=False, gradient_checkpointing=not args.tiny,
                     gradient_checkpointing_kwargs={"use_reentrant": False},
                     dataloader_pin_memory=False, remove_unused_columns=False, disable_tqdm=True,
                     mask_truncated_completions=False, use_vllm=False)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=cfg,
                          train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                          peft_config=LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                target_modules=targets, task_type="CAUSAL_LM"))
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
        with torch.no_grad():
            return policy(**probe).logits.detach().float().cpu()

    initial = logits()
    with policy.disable_adapter():
        ref_before = logits()

    def evaluate_actions(label):
        if args.tiny:
            return
        outcomes = []
        policy.eval()
        for row in checked[1]:
            visible = messages(restore(row["prefix"]).observation())
            prompt = tokenizer.apply_chat_template(visible, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(policy.device)
            if inputs["input_ids"].shape[1] + completion_length > model.config.max_position_embeddings:
                raise ValueError("evaluation context exceeds model budget")
            with torch.no_grad():
                output = policy.generate(**inputs, do_sample=False, max_new_tokens=completion_length,
                                         pad_token_id=tokenizer.pad_token_id)
            raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            _, result, progress = transition(row["prefix"], raw)
            outcomes.append(dict(id=row["id"], visible_input=visible, output=raw, transition=result, progress=progress))
        dump(args.output / f"{label}_development_eval.json", dict(scope="development_only_not_blind",
                                                                 outcomes=outcomes))

    dump(args.output / "manifest.json", dict(framework="TRL GRPOTrainer + PEFT", trl="0.26.2",
         source_sha256={p: digest(ROOT / p) for p in ("scripts/train_grpo.py", "characore/grpo_rewards.py", "characore/grpo_data.py", "characore/agent.py")},
         tiny=args.tiny, claim="software mechanics only" if args.tiny else "development next-action GRPO, not full-episode GRPO or held-out improvement",
         suite_sha256=None if args.tiny else digest(args.suite / "freeze.json"),
         calibration_report=None if args.tiny else checked[2],
         judge_metadata=None if args.tiny else judge_policy.metadata,
         base_files_sha256={p.name: digest(p) for p in sorted((args.output / "tiny_base" if args.tiny else Path(args.base)).iterdir()) if p.is_file()},
         seed=args.seed, group_size=group_size, steps=args.steps, config=cfg.to_dict(), reward_spec=REWARD_SPEC if not args.tiny else "synthetic token-ID mean",
         packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate")}))
    evaluate_actions("before")
    result = trainer.train()
    for hook in hooks:
        hook.remove()
    trained = logits()
    evaluate_actions("after")
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
                 changed_tensors=changed, gradients=grads, reference_logits_unchanged=True,
                 probe_logit_max_delta=(trained-initial).abs().max().item(),
                 reload_logit_max_error=(restored-trained).abs().max().item(),
                 adapter_sha256=digest(args.output / "adapter/adapter_model.safetensors"),
                 metrics=result.metrics, history=trainer.state.log_history)
    dump(args.output / "verification.json", proof)
    print(json.dumps(proof, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--judge-base")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--judge-results", type=Path)
    parser.add_argument("--human", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.tiny and (args.device != "cpu" or args.quantize or any((args.base,args.judge_base,args.suite,args.packet,args.human,args.judge_results))):
        parser.error("tiny mode is CPU-only random model; cannot mix real inputs")
    if not args.tiny and not all((args.base,args.judge_base,args.suite,args.packet,args.human,args.judge_results)):
        parser.error("real mode requires base, judge-base, suite, packet, judge-results and human")
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "invocation.json", {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    try:
        checked = preflight(args)
        if args.preflight_only:
            dump(args.output / "preflight.json", dict(passed=True, training_started=False, tiny=args.tiny))
        else:
            run(args, checked)
    except Exception:
        dump(args.output / "failure.json", dict(traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
