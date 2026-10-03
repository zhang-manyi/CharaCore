"""Held-out evaluation on the test templates: greedy replies from base (+ optional adapter).

Reports, for the evaluated model:
  win rate vs the frozen base replies (AB/BA, order-consistent pairs only),
  rule metrics (hard violations, style score, length, marker rate),
  cross-character rate: the judge is shown both cards and asked who said the
    reply, in both card orders; only order-consistent answers count,
  and a side-by-side table of base and evaluated replies.
Without --adapter the evaluated model is the base itself, which gives the
attribution and rule baseline (its win rate vs itself is a tie by construction).
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from characore.judge import make_attribution_request, parse_attribution
from characore.judge_runner import call_judge
from characore.persona import OTHER
from characore.protocol import digest, dump
from characore.style_data import load_base_replies, load_suite
from characore.style_rewards import PairJudge, hard_violation, pair_outcome, rule_metrics


def attribution(rows, replies, judge, output):
    """Who said it, asked twice with the card order swapped. Disagreement is reported, not resolved."""
    records = []
    for row in rows:
        reply = replies[row["id"]]
        if not reply:
            records.append(dict(id=row["id"], status="empty"))
            continue
        answers = []
        for reverse in (False, True):
            request = make_attribution_request(row, reply, reverse=reverse)
            request["id"] = f"{row['id']}_{'swap' if reverse else 'base'}"
            final = call_judge(request, judge, output / request["id"], retries=0,
                               parse=parse_attribution)["final"]
            answers.append(final["judgement"]["speaker"] if final["call_status"] == "ok" else None)
        if None in answers or answers[0] != answers[1]:
            records.append(dict(id=row["id"], status="inconsistent", answers=answers))
        else:
            records.append(dict(id=row["id"], status="ok", speaker=answers[0], character=row["character"]))
    ok = [r for r in records if r["status"] == "ok"]
    return dict(cases=len(records), consistent=len(ok),
                correct_rate=sum(r["speaker"] == r["character"] for r in ok) / len(ok) if ok else None,
                cross_character_rate=sum(r["speaker"] == OTHER[r["character"]] for r in ok) / len(ok) if ok else None,
                neither_rate=sum(r["speaker"] == "neither" for r in ok) / len(ok) if ok else None,
                records=records)


def win_rate(rows, replies, base, pairs):
    pending, keyed = {}, {}
    for row in rows:
        reply = replies[row["id"]]
        if hard_violation(reply, row["character"]) or reply == base[row["id"]]:
            continue
        reqs = pairs.requests(row, reply, base[row["id"]])
        keyed[row["id"]] = {label: key for label, (key, _) in reqs.items()}
        for key, request in reqs.values():
            pending.setdefault(key, request)
    calls = pairs.run(pending) if pending else {}
    records = []
    for row in rows:
        reply = replies[row["id"]]
        if hard_violation(reply, row["character"]):
            records.append(dict(id=row["id"], status="hard", reason=hard_violation(reply, row["character"]), score=0.0))
        elif row["id"] not in keyed:
            records.append(dict(id=row["id"], status="identical", score=0.5))
        else:
            try:
                score, off_topic, inconsistent = pair_outcome(calls[keyed[row["id"]]["AB"]],
                                                              calls[keyed[row["id"]]["BA"]])
                records.append(dict(id=row["id"], status="ok", score=score, off_topic=off_topic,
                                    order_inconsistent=inconsistent))
            except ValueError as exc:
                records.append(dict(id=row["id"], status="unusable", reason=str(exc)))
    scored = [r for r in records if r["status"] != "unusable"]
    return dict(cases=len(records), scored=len(scored),
                # A hard violation counts as a loss to base; identical replies and AB/BA disagreement as ties.
                win_rate=sum(r["score"] for r in scored) / len(scored) if scored else None,
                wins=sum(r["score"] == 1.0 for r in scored), ties=sum(r["score"] == 0.5 for r in scored),
                losses=sum(r["score"] == 0.0 for r in scored),
                order_inconsistent=sum(bool(r.get("order_inconsistent")) for r in scored),
                unusable=len(records) - len(scored), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-replies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--backend", choices=("api", "stub"), default="api")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    if args.backend == "api" and not args.allow_api:
        parser.error("API evaluation requires --allow-api")
    train, test = load_suite(args.suite)
    rows = test if args.split == "test" else train
    base = load_base_replies(args.base_replies, args.suite)
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import set_seed
    from characore.model_loader import load_model
    from characore.precision import select_precision
    from characore.style_generate import first_clean, generate

    set_seed(17)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = select_precision(device, args.precision)
    model, tokenizer = load_model(args.base, device=device, precision=precision)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.config.use_cache = True
    replies = first_clean(generate(model, tokenizer, rows, args.max_new_tokens, args.batch_size))
    dump(args.output / "replies.json", replies)
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    if args.backend == "api":
        from characore.api_judge import APIJudge
        judge = APIJudge(args.env_file, allow_calls=True)
    else:
        from characore.stub_judge import StubJudge
        judge = StubJudge()
    characters = [r["character"] for r in rows]
    report = dict(split=args.split, rows=len(rows), adapter=str(args.adapter) if args.adapter else None,
                  adapter_sha256=digest(args.adapter / "adapter_model.safetensors") if args.adapter else None,
                  judge=judge.metadata,
                  rules=dict(evaluated=rule_metrics([replies[r["id"]] for r in rows], characters),
                             base=rule_metrics([base[r["id"]] for r in rows], characters)),
                  vs_base=win_rate(rows, replies, base, PairJudge(judge, args.output / "pair_calls")),
                  attribution=attribution(rows, replies, judge, args.output / "attr_calls"))
    report["calls_made"] = judge.calls
    report["claim"] = ("judge-based metrics are uncalibrated: agreement with human judgement is unmeasured; "
                       "situations are original, not from the original work")
    dump(args.output / "report.json", report)
    table = ["| id | 对方台词 | base | 评测模型 |", "| --- | --- | --- | --- |"]
    for row in rows:
        cell = lambda s: s.replace("|", "\\|")
        table.append(f"| {row['id']} | {cell(row['line'])} | {cell(base[row['id']])} | {cell(replies[row['id']])} |")
    with (args.output / "side_by_side.md").open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(table) + "\n")
    print(json.dumps({k: report[k] for k in ("split", "rows", "rules", "calls_made")}, ensure_ascii=False))
    print(json.dumps(dict(win_rate=report["vs_base"]["win_rate"], unusable=report["vs_base"]["unusable"],
                          cross_character_rate=report["attribution"]["cross_character_rate"],
                          correct_rate=report["attribution"]["correct_rate"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
