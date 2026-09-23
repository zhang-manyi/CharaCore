"""One-decision environment rewards and calibrated pairwise LLM rewards for TRL."""
import json
import math
from pathlib import Path

from characore.agent import DeliveryTask, TASK_ID, dump, messages
from characore.judge import make_request
from characore.judge_runner import call_judge, identity

REWARD_VERSION = "agent-next-action-pairwise-v1"
REWARD_SPEC = dict(version=REWARD_VERSION, invalid_action=-1.0,
                   judge_weight=.8, progress_weight=.2,
                   pairwise_values={"A": 1.0, "B": 0.0, "tie": .5},
                   missing="abort_batch_no_update", equal_group="abort_batch_no_update",
                   unit="one_action_from_visible_task_checkpoint_not_full_episode")


class UnusableReward(ValueError):
    pass


def restore(prefix):
    # Same horizon as a real episode: a training checkpoint must produce the same
    # visible observation (including steps_remaining) that evaluation would.
    env = DeliveryTask()
    if not isinstance(prefix, list) or len(prefix) >= env.max_steps:
        raise ValueError("prefix must leave room for an action")
    for raw in prefix:
        if not isinstance(raw, str):
            raise ValueError("prefix entries must be raw JSON strings")
        row = env.step(raw)
        if row["failure"] or env.state["status"] != "active":
            raise ValueError("training checkpoint requires valid nonterminal history")
    return env


def transition(prefix, raw):
    env = restore(prefix)
    visible = messages(env.observation())
    row = env.step(raw)
    progress = row["tool_result"]["ok"] and any(
        key in row["state_changes"] for key in ("known_facts", "commitments", "location", "delivered"))
    return visible, row, float(progress)


def pair_reward(ab, ba):
    for result in (ab, ba):
        if result["call_status"] != "ok":
            raise UnusableReward("judge technical failure; never replace with zero")
        judgement = result["judgement"]
        if judgement["winner"] == "insufficient" or any(
                score["status"] == "insufficient" for dims in judgement["scores"].values() for score in dims.values()):
            raise UnusableReward("insufficient judge evidence; never replace with zero")
    winner = ab["judgement"]["winner"]
    reverse = {"A": "B", "B": "A", "tie": "tie"}[ba["judgement"]["winner"]]
    if winner != reverse:
        raise UnusableReward("judge order disagreement")
    return REWARD_SPEC["pairwise_values"][winner]


def validate_groups(values, group_size):
    """Reject unusable rewards. Equal-but-valid groups are reported, not rejected."""
    if group_size < 2 or not values or len(values) % group_size:
        raise UnusableReward("incomplete GRPO sampling group")
    degenerate = 0
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in group):
            raise UnusableReward("missing or nonfinite reward; abort before TRL nansum")
        # An all-equal group yields zero advantage for every sample: a no-op update,
        # not corrupt data. Aborting the batch would stall on easy checkpoints, so
        # count it instead of inventing a difference.
        degenerate += max(group) - min(group) < 1e-8
    return values, degenerate


class ActionReward:
    __name__ = "calibrated_action_reward"

    def __init__(self, judge_policy, output, rubric, group_size=2, distributed=False):
        self.judge = judge_policy
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.rubric = rubric
        self.group_size = group_size
        self.distributed = distributed
        self.batch = 0

    def __call__(self, completions, prefix, anchor, **kwargs):
        self.batch += 1
        path = self.output / f"batch_{self.batch:04d}"
        path.mkdir()
        values, audit = [], []
        try:
            if not len(completions) == len(prefix) == len(anchor):
                raise ValueError("reward batch alignment mismatch")
            if not self.distributed:
                # Under DDP a rank holds a partial group; DistributedReward checks
                # group composition after gathering every rank's keys.
                if len(completions) % self.group_size:
                    raise UnusableReward("incomplete GRPO sampling group")
                for start in range(0, len(completions), self.group_size):
                    keys = {identity(dict(prefix=prefix[i], anchor=anchor[i]))
                            for i in range(start, start + self.group_size)}
                    if len(keys) != 1:
                        raise UnusableReward("a GRPO group must share one visible checkpoint and anchor")
            for i, (raw, history, reference) in enumerate(zip(completions, prefix, anchor)):
                # The trainer uses pre-rendered string prompts, so completions are strings.
                visible, row, progress = transition(history, raw)
                sample = dict(output=raw, transition=row, progress=progress)
                audit.append(sample)
                dump(path / f"sample_{i:03d}.json", sample)
                if row["failure"]:
                    # Policy errors are real negative outcomes; judge infrastructure failures are not.
                    values.append(-1.0)
                    continue
                _, anchor_row, _ = transition(history, reference)
                candidates = {"A": json.dumps(dict(output=raw, execution=row["tool_result"]), ensure_ascii=False),
                              "B": json.dumps(dict(output=reference, execution=anchor_row["tool_result"]), ensure_ascii=False)}
                if candidates["A"] == candidates["B"]:
                    # The policy reproduced the reference action. Identical candidates
                    # make any winner an order artifact, so score it a tie without
                    # spending a judge call.
                    values.append(REWARD_SPEC["judge_weight"] * REWARD_SPEC["pairwise_values"]["tie"]
                                  + REWARD_SPEC["progress_weight"] * progress)
                    continue
                context = dict(id=TASK_ID, character="岚（原创设计角色，守诺、保护档案）", action_required=True,
                               visible_turns=[dict(source_line=1, speaker="可见任务输入", text=visible[1]["content"])])
                results = []
                for reverse in (False, True):
                    request = make_request(context, candidates, self.rubric, reverse=reverse)
                    request["id"] = f"sample_{i:03d}_{'BA' if reverse else 'AB'}"
                    results.append(call_judge(request, self.judge, path / request["id"])["final"])
                values.append(REWARD_SPEC["judge_weight"] * pair_reward(*results)
                              + REWARD_SPEC["progress_weight"] * progress)
            degenerate = None
            if not self.distributed:
                _, degenerate = validate_groups(values, self.group_size)
            dump(path / "rewards.json", dict(values=values, spec=REWARD_SPEC,
                                             zero_advantage_groups=degenerate))
            return values
        except Exception as exc:
            dump(path / "failure.json", dict(type=type(exc).__name__, message=str(exc), partial_rewards=values))
            raise
