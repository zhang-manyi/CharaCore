"""Measure how many distinct actions the policy samples per frozen training checkpoint.

GRPO only learns from groups whose rewards differ. grpo_05 aborted on its first
batch with every sample in a group byte-identical, so before changing the reward
or the framework this measures the sampling distribution itself: same prompt
construction and sampling settings as scripts/train_grpo.py, no judge call, no
API, no parameter update. The reference action never enters the prompt.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.agent import dump, messages, parse_output
from characore.grpo_data import load_training_suite
from characore.grpo_rewards import restore, transition


def action_key(raw):
    """Canonical tool call, ignoring speech and JSON whitespace; None if unparsable."""
    try:
        parsed = parse_output(raw)
    except (ValueError, TypeError):
        return None
    return json.dumps(dict(tool=parsed["tool"], arguments=parsed["arguments"]),
                      ensure_ascii=False, sort_keys=True)


def canonical(raw):
    try:
        return json.dumps(parse_output(raw), ensure_ascii=False, sort_keys=True)
    except (ValueError, TypeError):
        return None


def summarize(row, raws):
    """Per-checkpoint diversity. Environment-only contrast needs no judge to separate the group."""
    samples = []
    for raw in raws:
        _, result, progress = transition(row["prefix"], raw)
        samples.append(dict(output=raw, action=action_key(raw),
                            failure=result["failure"]["kind"] if result["failure"] else None,
                            progress=progress))
    anchor_action, anchor_full = action_key(row["anchor"]), canonical(row["anchor"])
    outcomes = {(s["failure"], s["progress"]) for s in samples}
    return dict(id=row["id"], depth=len(row["prefix"]), samples=len(raws),
                distinct_outputs=len(set(raws)),
                distinct_actions=len({s["action"] for s in samples}),
                invalid=sum(s["failure"] is not None for s in samples),
                anchor_action=sum(s["action"] == anchor_action for s in samples),
                # Same speech and call as the anchor but different bytes: the reward's
                # byte comparison still spends judge calls on these.
                anchor_equivalent_not_bytes=sum(canonical(s["output"]) == anchor_full and s["output"] != row["anchor"]
                                                for s in samples),
                environment_contrast=len(outcomes) > 1,
                action_counts=Counter(str(s["action"]) for s in samples).most_common(),
                outputs=samples)


def overall(records):
    def share(key):
        return sum(bool(r[key]) for r in records)
    by_depth = {}
    for r in records:
        d = by_depth.setdefault(str(r["depth"]), dict(checkpoints=0, one_output=0, one_action=0))
        d["checkpoints"] += 1
        d["one_output"] += r["distinct_outputs"] == 1
        d["one_action"] += r["distinct_actions"] == 1
    return dict(checkpoints=len(records),
                one_output=sum(r["distinct_outputs"] == 1 for r in records),
                one_action=sum(r["distinct_actions"] == 1 for r in records),
                environment_contrast=share("environment_contrast"),
                all_anchor_action=sum(r["anchor_action"] == r["samples"] for r in records),
                samples_anchor_equivalent_not_bytes=sum(r["anchor_equivalent_not_bytes"] for r in records),
                by_depth=by_depth,
                claim="sampling diversity only; no reward, no judge, no training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    # Defaults mirror the GRPOConfig in scripts/train_grpo.py.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.samples < 2 or args.temperature <= 0 or args.max_new_tokens < 1:
        parser.error("need samples >= 2, temperature > 0, max-new-tokens >= 1")
    train, _ = load_training_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=False)

    import importlib.metadata
    import torch
    from transformers import GenerationConfig, set_seed
    from characore.model_loader import load_model
    from characore.precision import select_precision

    set_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = select_precision(device, args.precision)
    model, tokenizer = load_model(args.base, device=device, precision=precision)
    model.eval()
    model.config.use_cache = True
    # Built the same way as TRL's GRPOTrainer. Recent Transformers may fill fields
    # left at global defaults (temperature 1.0, top_p 1.0) from the checkpoint's
    # generation_config.json, so the checkpoint config is recorded alongside.
    generation = GenerationConfig(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                  max_new_tokens=args.max_new_tokens, num_return_sequences=args.samples,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    dump(args.output / "manifest.json", dict(
        base=str(args.base), device=device, precision=precision, seed=args.seed,
        suite_freeze=str(args.suite / "freeze.json"), generation=generation.to_dict(),
        checkpoint_generation_config=model.generation_config.to_dict(),
        packages={p: importlib.metadata.version(p) for p in ("torch", "transformers")}))
    records = []
    for row in train:
        prompt = tokenizer.apply_chat_template(messages(restore(row["prefix"]).observation()),
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        width = inputs["input_ids"].shape[1]
        if width + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("prompt too long; refusing silent truncation")
        with torch.inference_mode():
            out = model.generate(**inputs, generation_config=generation)
        raws = [tokenizer.decode(ids[width:], skip_special_tokens=True) for ids in out]
        record = summarize(row, raws)
        records.append(record)
        print(json.dumps({k: record[k] for k in ("id", "depth", "distinct_outputs", "distinct_actions",
                                                 "invalid", "anchor_action")}), flush=True)
    dump(args.output / "checkpoints.json", records)
    summary = overall(records)
    summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device != "cpu" else None
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
