"""Explicit frozen inputs and calibration gates; no bundled training data."""
from pathlib import Path

from characore.agent import TASK_ID, messages
from characore.calibration_report import read, summarize
from characore.grpo_rewards import REWARD_SPEC, restore, transition
from characore.judge_runner import identity
from characore.protocol import digest


def load_training_suite(directory):
    directory = Path(directory)
    freeze = read(directory / "freeze.json")
    required = {"train.json", "eval.json", "reward_approval.json"}
    if set(freeze["files"]) != required or freeze.get("task_id") != TASK_ID:
        raise ValueError("explicit frozen task train/eval/reward approval required")
    scope = freeze.get("evaluation_scope")
    if scope not in ("development_only", "independent_holdout"):
        raise ValueError("evaluation_scope must explicitly distinguish development from holdout")
    for name in required:
        if digest(directory / name) != freeze["files"][name]:
            raise ValueError(f"frozen file changed: {name}")
    train, evaluation = read(directory / "train.json"), read(directory / "eval.json")
    if not train or not evaluation:
        raise ValueError("nonempty train and independent evaluation required")
    ids, observations, groups = set(), set(), {"train": set(), "eval": set()}
    for split, rows in (("train", train), ("eval", evaluation)):
        for row in rows:
            if row["id"] in ids or row["split"] != split:
                raise ValueError("duplicate ID or mismatched split")
            if row.get("exposure") == "development" and split == "eval" and scope != "development_only":
                raise ValueError("development task cannot be relabeled held-out evaluation")
            if not row.get("family") or not row.get("source"):
                raise ValueError("source and task family required")
            # All current authored task checkpoints share one source family.
            if row["family"] != TASK_ID or row["source"] != TASK_ID:
                raise ValueError("current environment has one family; renaming does not create a holdout")
            ids.add(row["id"])
            groups[split].add(row["family"])
            env = restore(row["prefix"])
            visible = messages(env.observation())
            signature = identity(visible)
            if signature in observations:
                raise ValueError("duplicate visible training/evaluation checkpoint")
            observations.add(signature)
            _, result, _ = transition(row["prefix"], row["anchor"])
            if result["failure"]:
                raise ValueError("anchor must be an executable action, not a gold label")
    if scope == "independent_holdout" and groups["train"] & groups["eval"]:
        raise ValueError("task-family leakage: current single development task has no independent holdout")
    approval = read(directory / "reward_approval.json")
    if (approval.get("reward_spec_sha256") != identity(REWARD_SPEC) or not approval.get("reviewer")
            or not approval.get("completed_at") or approval.get("approved") is not True):
        raise ValueError("actual human reward approval bound to exact reward spec required")
    return train, evaluation, approval


def verify_calibration(packet, results, human):
    report = summarize(packet, results, human)
    if not report["calibration_passed"]:
        raise ValueError("actual human/model calibration has not passed")
    if report["scope"] != TASK_ID:
        raise ValueError("Genshin calibration cannot authorize the authored Agent task")
    return report
