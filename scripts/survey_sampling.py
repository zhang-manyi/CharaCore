"""Survey the untrained base before GRPO: within-group diversity and the starting win rate vs base.

GRPO only learns from groups whose rewards differ, and a policy that already
beats (or always loses to) the frozen base reply leaves little signal. Samples
use the same prompt and sampling settings as scripts/train_grpo.py; no
parameter update. Without a judge it reports diversity and rule scores only;
with --judge-backend stub|api it scores --judge-rows rows through the real
StyleReward, so its rejection and reuse counts match training.

Gate (written to summary.json, decision left to the operator):
  at least 60% of judged groups have non-identical rewards, and
  the starting win rate vs base lies in [0.1, 0.9].
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.protocol import dump
from characore.style_data import load_base_replies, load_suite
from characore.style_rewards import clean_reply, hard_violation, style_score

GATE = dict(min_varied_share=0.6, win_rate_range=(0.1, 0.9))


def summarize(row, raws):
    """Per-row diversity on cleaned replies; rule scores need no judge."""
    replies = [clean_reply(r) for r in raws]
    styles = [style_score(r, row["character"]) for r in replies]
    hard = [hard_violation(r, row["character"]) for r in replies]
    return dict(id=row["id"], character=row["character"], samples=len(replies),
                distinct_replies=len(set(replies)), distinct_style_scores=len(set(styles)),
                hard=sum(h is not None for h in hard), hard_reasons=sorted({h for h in hard if h}),
                style_mean=sum(styles) / len(styles), replies=replies)


def overall(records, judged=None):
    n = len(records)
    out = dict(rows=n,
               one_reply=sum(r["distinct_replies"] == 1 for r in records),
               all_hard=sum(r["hard"] == r["samples"] for r in records),
               hard_rate=sum(r["hard"] for r in records) / max(sum(r["samples"] for r in records), 1),
               style_mean=sum(r["style_mean"] for r in records) / n if n else None,
               by_character={c: sum(r["character"] == c for r in records) for c in ("rei", "asuka")},
               claim="sampling diversity and rule scores; win rate only if a judge ran")
    if judged:
        usable = judged["groups"] - judged["rejected_groups"]
        varied_share = judged["varied_groups"] / usable if usable else None
        win = judged["win_rate_vs_base"]
        out.update(judged=judged, varied_share=varied_share,
                   gate=dict(GATE, passed=varied_share is not None and varied_share >= GATE["min_varied_share"]
                             and win is not None and GATE["win_rate_range"][0] <= win <= GATE["win_rate_range"][1]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-replies", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    # Defaults mirror the GRPOConfig in scripts/train_grpo.py.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--judge-backend", choices=("none", "stub", "api"), default="none")
    parser.add_argument("--judge-rows", type=int, default=20)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.samples < 2 or args.temperature <= 0 or args.max_new_tokens < 1:
        parser.error("need samples >= 2, temperature > 0, max-new-tokens >= 1")
    if args.judge_backend != "none" and args.base_replies is None:
        parser.error("judging needs --base-replies")
    if args.judge_backend == "api" and not args.allow_api:
        parser.error("API judging requires --allow-api")
    train, _ = load_suite(args.suite)
    base_replies = load_base_replies(args.base_replies, args.suite) if args.base_replies else None
    args.output.mkdir(parents=True, exist_ok=False)

    import importlib.metadata
    import torch
    from transformers import GenerationConfig, set_seed
    from characore.model_loader import load_model
    from characore.precision import select_precision
    from characore.style_generate import generate

    set_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = select_precision(device, args.precision)
    model, tokenizer = load_model(args.base, device=device, precision=precision)
    model.config.use_cache = True
    generation = GenerationConfig(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                  max_new_tokens=args.max_new_tokens, num_return_sequences=args.samples,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    dump(args.output / "manifest.json", dict(
        base=str(args.base), device=device, precision=precision, seed=args.seed,
        suite_freeze=str(args.suite / "freeze.json"), generation=generation.to_dict(),
        checkpoint_generation_config=model.generation_config.to_dict(), judge_backend=args.judge_backend,
        packages={p: importlib.metadata.version(p) for p in ("torch", "transformers")}))
    raws = generate(model, tokenizer, train, args.max_new_tokens, args.batch_size, generation)
    records = [summarize(row, raws[row["id"]]) for row in train]
    dump(args.output / "rows.json", records)

    judged = None
    if args.judge_backend != "none":
        from characore.style_rewards import StyleReward
        if args.judge_backend == "api":
            from characore.api_judge import APIJudge
            judge = APIJudge(args.env_file, allow_calls=True)
        else:
            from characore.stub_judge import StubJudge
            judge = StubJudge()
        chosen = random.Random(args.seed).sample(train, min(args.judge_rows, len(train)))
        reward = StyleReward(judge, args.output / "judge", args.samples)
        expand = lambda key: [r[key] for r in chosen for _ in range(args.samples)]
        reward(prompts=expand("id"), completions=[c for r in chosen for c in raws[r["id"]]],
               row_id=expand("id"), character=expand("character"),
               base_reply=[base_replies[r["id"]] for r in chosen for _ in range(args.samples)],
               situation=expand("situation"), speaker=expand("speaker"), line=expand("line"))
        judged = dict(reward.last_metrics, judge=judge.metadata, calls_made=judge.calls)
    summary = overall(records, judged)
    summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device != "cpu" else None
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
