"""Synchronize reward failures and validate complete groups before TRL aggregation."""
from characore.grpo_rewards import UnusableReward, validate_groups


def check_payloads(payloads, group_size):
    if any(p["error"] for p in payloads):
        raise UnusableReward("at least one rank failed reward calculation; all ranks abort before update")
    values = [v for p in payloads for v in p["values"]]
    keys = [v for p in payloads for v in p["keys"]]
    if len(keys) != len(values):
        raise UnusableReward("reward group key alignment mismatch")
    validate_groups(values, group_size)
    for start in range(0, len(keys), group_size):
        if len(set(keys[start:start + group_size])) != 1:
            raise UnusableReward("distributed GRPO group contains different prompts/anchors")


class DistributedReward:
    __name__ = "synchronized_reward"

    def __init__(self, callback, group_size):
        self.callback = callback
        self.group_size = group_size

    def __call__(self, **kwargs):
        import torch.distributed as dist
        from characore.judge_runner import identity
        prompts = kwargs["prompts"]
        prefix, anchor = kwargs.get("prefix"), kwargs.get("anchor")
        keys = [identity(dict(prompt=p, prefix=prefix[i] if prefix is not None else None,
                              anchor=anchor[i] if anchor is not None else None)) for i, p in enumerate(prompts)]
        payload = dict(values=[], keys=keys, error=None)
        try:
            payload["values"] = self.callback(**kwargs)
            if len(payload["values"]) != len(keys):
                raise ValueError("reward callback length mismatch")
        except Exception as exc:
            payload["error"] = type(exc).__name__
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        check_payloads(gathered, self.group_size)
        return payload["values"]
