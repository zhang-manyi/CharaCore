"""Measure judge AB/BA order consistency and repeat agreement on frozen base replies.

Training rejects any group with an order disagreement, so a position-biased
judge stalls GRPO regardless of GPU or schema correctness. grpo_05 also showed
the judge can return different verdicts for byte-identical requests, which is
why training reuses one call per distinct request; this script measures how
often that happens. No model, no training, no parameter update.

Per case (one suite row of character c): candidate A is c's frozen base reply,
candidate B is the other character's frozen base reply to the same situation
and line, judged under c's card. Three calls: AB, BA, and AB again.
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.canon import canon_identity, load_canon
from characore.judge import make_request
from characore.judge_runner import call_judge
from characore.persona import OTHER
from characore.protocol import dump
from characore.style_data import load_base_replies, load_suite
from characore.style_rewards import pair_outcome


def winner(record):
    final = record["final"]
    if final["call_status"] != "ok":
        return final["call_status"]
    w = final["judgement"]["winner"]
    return record["mapping"][w] if w in ("A", "B") else w


def probe(row, candidates, judge, output, canon=None):
    calls = {}
    for label, reverse in (("AB", False), ("BA", True), ("AB_repeat", False)):
        request = make_request(row, candidates, reverse=reverse, canon=canon)
        request["id"] = f"{row['id']}_{label}"
        calls[label] = call_judge(request, judge, output / request["id"], retries=0)
        calls[label]["mapping"] = request["display_to_original"]
    record = dict(id=row["id"], character=row["character"],
                  **{f"{k}_winner": winner(v) for k, v in calls.items()},
                  repeat_agrees=winner(calls["AB"]) == winner(calls["AB_repeat"]))
    try:
        outcome = pair_outcome(calls["AB"], calls["BA"])
        # The reward averages the two orders' scores, so the per-order margin gap is the order noise it carries.
        record.update(margin=outcome["margin"], order_margin_gap=abs((outcome["policy"][0] - outcome["base"][0])
                                                                     - (outcome["policy"][1] - outcome["base"][1])))
        if outcome["order_inconsistent"]:
            # The verdict is logged only now, but a position-following verdict is still the failure measured here.
            raise ValueError("order_inconsistent")
        record.update(status="usable", own_base_score=outcome["verdict"], off_topic=outcome["off_topic"])
    except ValueError as exc:
        # The exact reason training would have rejected a group containing this pair.
        record.update(status="unusable", reason=str(exc))
    return record


def summarize(records, judge):
    n = len(records)
    usable = [r for r in records if r["status"] == "usable"]
    margined = [r for r in records if "margin" in r]
    reasons = {}
    for r in records:
        if r["status"] == "unusable":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return dict(cases=n, usable=len(usable), usable_rate=round(len(usable) / n, 3) if n else None,
                repeat_agreement=round(sum(r["repeat_agrees"] for r in records) / n, 3) if n else None,
                own_character_preferred=round(sum(r["own_base_score"] for r in usable) / len(usable), 3) if usable else None,
                own_character_margin=round(sum(r["margin"] for r in margined) / len(margined), 3) if margined else None,
                order_margin_gap_mean=round(sum(r["order_margin_gap"] for r in margined) / len(margined), 3) if margined else None,
                reasons=reasons, judge=judge.metadata, calls_made=judge.calls,
                claim="order consistency and repeat agreement only; agreement with human judgement is unmeasured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-replies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("api", "stub"), default="api")
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--canon", type=Path, help="frozen canon reference set shown to the judge as evidence E4")
    args = parser.parse_args()
    if args.backend == "api" and not args.allow_api:
        parser.error("API measurement requires --allow-api")
    if args.cases < 1:
        parser.error("--cases must be positive")
    train, _ = load_suite(args.suite)
    replies = load_base_replies(args.base_replies, args.suite)
    canon = load_canon(args.canon) if args.canon else None
    if args.backend == "api":
        from characore.api_judge import APIJudge
        judge = APIJudge(args.env_file, allow_calls=True)
    else:
        from characore.stub_judge import StubJudge
        judge = StubJudge()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in random.Random(args.seed).sample(train, min(args.cases, len(train))):
        other_id = row["id"].rsplit("-", 1)[0] + "-" + OTHER[row["character"]]
        candidates = {"A": replies[row["id"]], "B": replies[other_id]}
        if candidates["A"] == candidates["B"] or not all(candidates.values()):
            records.append(dict(id=row["id"], status="skipped", reason="identical or empty base replies"))
            continue
        record = probe(row, candidates, judge, args.output / "calls", canon)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    scored = [r for r in records if r["status"] != "skipped"]
    summary = summarize(scored, judge)
    summary["skipped"] = len(records) - len(scored)
    summary["canon_sha256"] = canon_identity(args.canon)
    dump(args.output / "cases.json", records)
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if scored and summary["usable"] == len(scored) else 2


if __name__ == "__main__":
    raise SystemExit(main())
