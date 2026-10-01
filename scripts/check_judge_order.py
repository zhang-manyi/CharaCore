"""Measure judge AB/BA order self-consistency on frozen checkpoints.

pair_reward rejects a sample whenever the AB and BA calls disagree, and
REWARD_SPEC aborts the whole batch on one unusable reward, so a judge with a
position bias stalls training regardless of GPU or schema correctness. This
script measures that rate directly: no model, no training, no parameter update.

Candidate A is the recorded anchor action; candidate B is a perturbed variant
whose speech differs while the tool call stays identical, which is the shape
the policy actually produces. A self-consistent judge returns mirrored winners
for the two orders; anything else is what aborts a batch.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.agent import dump
from characore.grpo_data import load_training_suite
from characore.grpo_rewards import UnusableReward, pair_reward, transition
from characore.judge import make_request
from characore.judge_runner import call_judge
from characore.protocol import read_json

def variant(anchor):
    """Reword the speech, keep the tool call. Same shape as a sampled completion."""
    row = json.loads(anchor)
    row["speech"] = "我会守住承诺，保持封条完整，再继续下一步。"
    return json.dumps(row, ensure_ascii=False)


def probe(row, judge, rubric, output):
    """One checkpoint, two orders. Returns the pair outcome without inventing a reward."""
    from characore.agent import TASK_ID

    reworded = variant(row["anchor"])
    visible, candidate_row, progress = transition(row["prefix"], reworded)
    _, anchor_row, _ = transition(row["prefix"], row["anchor"])
    candidates = {"A": json.dumps(dict(output=reworded, execution=candidate_row["tool_result"]),
                                  ensure_ascii=False),
                  "B": json.dumps(dict(output=row["anchor"], execution=anchor_row["tool_result"]),
                                  ensure_ascii=False)}
    if candidate_row["failure"] or candidates["A"] == candidates["B"]:
        return dict(id=row["id"], status="skipped",
                    note="perturbation failed execution or matched the anchor byte for byte")
    context = dict(id=TASK_ID, character="岚（原创设计角色，守诺、保护档案）", action_required=True,
                   visible_turns=[dict(source_line=1, speaker="可见任务输入",
                                       text=visible[1]["content"])])
    calls = {}
    for reverse in (False, True):
        request = make_request(context, candidates, rubric, reverse=reverse)
        label = "BA" if reverse else "AB"
        request["id"] = f"{row['id']}_{label}"
        calls[label] = call_judge(request, judge, output / request["id"])["final"]
    record = dict(id=row["id"],
                  ab=calls["AB"]["call_status"], ba=calls["BA"]["call_status"],
                  ab_winner=(calls["AB"]["judgement"] or {}).get("winner"),
                  ba_winner=(calls["BA"]["judgement"] or {}).get("winner"),
                  ab_error=calls["AB"].get("error"), ba_error=calls["BA"].get("error"))
    try:
        record.update(status="usable", reward=pair_reward(calls["AB"], calls["BA"]))
    except UnusableReward as exc:
        # The exact reason training would have aborted on this checkpoint.
        record.update(status="unusable", reason=str(exc))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("api", "stub"), default="api")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if args.backend == "api" and not args.allow_api:
        parser.error("API measurement requires --allow-api")
    if args.cases < 1:
        parser.error("--cases must be positive")
    train, _ = load_training_suite(args.suite)
    if args.backend == "api":
        from characore.api_judge import APIJudge
        judge = APIJudge(args.env_file, allow_calls=args.allow_api)
    else:
        from characore.stub_judge import StubJudge
        judge = StubJudge()
    rubric = read_json(ROOT / "experiments/agent_v1/evaluation_plan.json")["rubric"]
    args.output.mkdir(parents=True, exist_ok=False)
    # Spread the sample across prefix depths instead of taking the first N rows,
    # which all share the shortest histories.
    step = max(len(train) // args.cases, 1)
    selected = train[::step][:args.cases]
    records = []
    for row in selected:
        record = probe(row, judge, rubric, args.output / "calls")
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    scored = [r for r in records if r["status"] != "skipped"]
    usable = [r for r in scored if r["status"] == "usable"]
    summary = dict(cases=len(records), scored=len(scored), usable=len(usable),
                   usable_rate=round(len(usable) / len(scored), 3) if scored else None,
                   judge=judge.metadata, calls_made=judge.calls,
                   reasons=sorted({r["reason"] for r in scored if r["status"] == "unusable"}),
                   claim="order self-consistency only; agreement with human judgement is still unmeasured")
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if scored and len(usable) == len(scored) else 2


if __name__ == "__main__":
    raise SystemExit(main())
