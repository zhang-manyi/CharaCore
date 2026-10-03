"""Style reward: rule gates, a capped rule style score, and a pairwise judge against frozen base replies.

Per sample:
  hard violation (out of character, length, catchphrase cap, impersonation,
  multi-line)                         -> -1, no judge call
  byte-identical to the base reply    -> pairwise 0.5 (tie), no judge call
  otherwise                           -> AB/BA judge calls against the base reply
  reward = 0.7 * pairwise + 0.3 * style_score - 0.5 * off_topic

pairwise is 1 / 0.5 / 0 for policy win / tie / base win, and counts only when
both orders agree. off_topic means both orders scored the policy reply's R <= 1.

A group that contains any unusable sample (judge failure, parse error, order
disagreement, insufficient) is rejected as a whole: every member returns None,
so TRL's nansum gives the group all-zero rewards and zero advantages, and the
trainer additionally zeroes its completion_mask. A partial None would be summed
as 0 and silently corrupt the group's mean and std, so it is never returned.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from characore.judge import PROTOCOL, make_request
from characore.judge_runner import call_judge, identity
from characore.persona import CHARACTERS, OTHER
from characore.protocol import dump

LENGTH = (1, 80)  # Rei's in-character "不。" is one character; per-persona brevity is the soft length term
# Any one tone marker more than this many times, or this many marker hits in total, is stuffing.
MARKER_CAP, MARKER_TOTAL_CAP = 2, 4
OOC = ("作为AI", "作为一个AI", "作为人工智能", "人工智能", "语言模型", "AI助手", "我是AI",
       "扮演", "角色设定", "设定中", "这个角色", "台词：", "旁白")
WEIGHTS = dict(pairwise=0.7, style=0.3, off_topic=0.5)
REWARD_SPEC = dict(name="style-pairwise-v1", judge_protocol=PROTOCOL, length=LENGTH,
                   marker_cap=MARKER_CAP, marker_total_cap=MARKER_TOTAL_CAP, ooc=OOC,
                   ellipsis="'...' and '…' runs count as '……' for markers only; reply text is unchanged",
                   weights=WEIGHTS, hard_penalty=-1.0,
                   rejection="any unusable sample rejects its whole group; never zero-filled")
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")
ELLIPSIS = re.compile(r"\.{3,}|…+")


def clean_reply(raw):
    """Strip the empty think block some templates emit and surrounding whitespace only."""
    return EMPTY_THINK.sub("", raw).strip()


def marker_text(reply):
    """The base model writes '...' about as often as '……'; both are the same tone marker."""
    return ELLIPSIS.sub("……", reply)


def hard_violation(reply, character):
    """Reason string for a reply that gets -1 without a judge call, else None."""
    person, other = CHARACTERS[character], CHARACTERS[OTHER[character]]
    if not LENGTH[0] <= len(reply) <= LENGTH[1]:
        return "length"
    if "\n" in reply:
        return "multi_line"
    if any(word in reply for word in OOC):
        return "out_of_character"
    if any(word in reply for word in person["foreign"]) or any(f"我是{a}" in reply for a in other["aliases"]):
        return "impersonation"
    # A speaker prefix for anyone ("明日香：…") is a script line, not a reply in character.
    if re.match(r"^[^，。！？\s]{1,6}[：:]", reply) and any(reply.startswith(a) for a in person["aliases"] + other["aliases"]):
        return "speaker_prefix"
    counts = [marker_text(reply).count(m) for m in person["markers"]]
    if max(counts) > MARKER_CAP or sum(counts) > MARKER_TOTAL_CAP:
        return "catchphrase_cap"
    return None


def style_score(reply, character):
    """Rule style score in [0, 1]. Presence counts once per marker, so repetition earns nothing."""
    person, text = CHARACTERS[character], marker_text(reply)
    present = sum(m in text for m in person["markers"])
    marker = min(present, 2) / 2
    anti = 0.0 if any(m in text for m in person["anti_markers"]) else 1.0
    n, spec = len(reply), person["length"]
    if n < spec["soft_min"]:
        length = 0.5
    elif n <= spec["soft_max"]:
        length = 1.0
    else:
        length = max(0.0, 1 - (n - spec["soft_max"]) / (spec["zero_at"] - spec["soft_max"]))
    return round(0.4 * marker + 0.3 * anti + 0.3 * length, 6)


def rule_metrics(replies, characters):
    """Batch rule metrics for curves; counts, not judgements."""
    n = len(replies)
    hard = [hard_violation(r, c) for r, c in zip(replies, characters)]
    reasons = {}
    for h in hard:
        if h:
            reasons[h] = reasons.get(h, 0) + 1
    return dict(samples=n, hard_violation_rate=sum(h is not None for h in hard) / n if n else None,
                hard_reasons=reasons,
                style_score_mean=sum(style_score(r, c) for r, c in zip(replies, characters)) / n if n else None,
                length_mean=sum(len(r) for r in replies) / n if n else None,
                marker_rate=sum(any(m in marker_text(r) for m in CHARACTERS[c]["markers"]) for r, c in zip(replies, characters)) / n if n else None)


def _original(final, mapping):
    """Translate a displayed winner and the policy's R score back to original labels."""
    judgement = final["judgement"]
    winner = judgement["winner"]
    original = mapping[winner] if winner in ("A", "B") else winner
    policy_side = next(side for side, orig in mapping.items() if orig == "A")
    r = judgement["scores"][policy_side]["R"]
    return original, (r["score"] if r["status"] == "scored" else None)


def pair_outcome(ab, ba):
    """Combine AB and BA results. Returns (pairwise, off_topic) or raises ValueError with the reason."""
    for label, call in (("AB", ab), ("BA", ba)):
        if call["final"]["call_status"] != "ok":
            raise ValueError(f"{label} {call['final']['call_status']}")
    first, r_first = _original(ab["final"], ab["mapping"])
    second, r_second = _original(ba["final"], ba["mapping"])
    if "insufficient" in (first, second):
        raise ValueError("insufficient")
    if first != second:
        raise ValueError("order_inconsistent")
    pairwise = {"A": 1.0, "tie": 0.5, "B": 0.0}[first]
    off_topic = r_first is not None and r_second is not None and r_first <= 1 and r_second <= 1
    return pairwise, off_topic


class PairJudge:
    """AB/BA judge calls with in-batch reuse of byte-identical requests."""

    def __init__(self, judge, output, workers=8, retries=0):
        self.judge, self.output, self.workers, self.retries = judge, Path(output), workers, retries
        self.output.mkdir(parents=True, exist_ok=False)
        self.batches = 0

    def requests(self, row, reply, base):
        out = {}
        for reverse in (False, True):
            request = make_request(row, {"A": reply, "B": base}, reverse=reverse)
            key = identity(request["messages"])
            request["id"] = key[:16]
            out["BA" if reverse else "AB"] = (key, request)
        return out

    def run(self, pending):
        """pending: {key: request}. One call per distinct key; returns {key: call record}."""
        self.batches += 1
        root = self.output / f"batch_{self.batches:05d}"

        def one(item):
            key, request = item
            record = call_judge(request, self.judge, root / key[:16], retries=self.retries)
            record["mapping"] = request["display_to_original"]
            return key, record

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(one, pending.items()))


class StyleReward:
    """TRL reward function. Single process: the batch holds whole consecutive groups of G."""

    def __init__(self, judge, output, group_size, workers=8, max_all_rejected=3):
        self.__name__ = "style"  # TRL names the reward column after this
        self.pairs = PairJudge(judge, Path(output) / "calls", workers)
        self.output = Path(output)
        self.group_size = group_size
        self.max_all_rejected = max_all_rejected
        self.all_rejected_streak = 0
        self.pending_rejected = None
        self.last_metrics = None
        self.totals = dict(batches=0, groups=0, rejected_groups=0, judge_calls=0, reused_requests=0,
                           skipped_identical=0, hard=0)

    def take_rejected(self):
        """Group indices rejected in the latest call. The trainer must consume exactly one per batch."""
        if self.pending_rejected is None:
            raise RuntimeError("reward was not computed for this batch")
        rejected, self.pending_rejected = self.pending_rejected, None
        return rejected

    def __call__(self, prompts, completions, completion_ids=None, row_id=None, character=None,
                 base_reply=None, situation=None, speaker=None, line=None, **kwargs):
        g = self.group_size
        n = len(completions)
        if n % g or not (len(row_id) == len(character) == len(base_reply) == n):
            raise ValueError("reward batch is not whole groups")
        for start in range(0, n, g):
            if len(set(row_id[start:start + g])) != 1 or len(set(prompts[start:start + g])) != 1:
                raise ValueError("group members must share one prompt; ordering assumption broken")
        replies = [clean_reply(c) for c in completions]
        rows = [dict(character=character[i], situation=situation[i], speaker=speaker[i], line=line[i])
                for i in range(n)]
        samples, pending, wanted = [], {}, 0
        for i, reply in enumerate(replies):
            sample = dict(row_id=row_id[i], reply=reply, hard=hard_violation(reply, character[i]),
                          style=style_score(reply, character[i]))
            if sample["hard"] is None and reply != base_reply[i]:
                reqs = self.pairs.requests(rows[i], reply, base_reply[i])
                sample["keys"] = {label: key for label, (key, _) in reqs.items()}
                for key, request in reqs.values():
                    wanted += 1
                    pending.setdefault(key, request)
            samples.append(sample)
        calls = self.pairs.run(pending) if pending else {}
        self.totals["judge_calls"] += len(calls)
        self.totals["reused_requests"] += wanted - len(pending)

        values, rejected, reasons = [None] * n, [], {}
        for group, start in enumerate(range(0, n, g)):
            members, error = [], None
            for i in range(start, start + g):
                s = samples[i]
                if s["hard"]:
                    s.update(pairwise=None, off_topic=False, value=-1.0)
                    self.totals["hard"] += 1
                elif "keys" not in s:
                    s.update(pairwise=0.5, off_topic=False, identical_to_base=True)
                    self.totals["skipped_identical"] += 1
                else:
                    try:
                        s["pairwise"], s["off_topic"] = pair_outcome(calls[s["keys"]["AB"]], calls[s["keys"]["BA"]])
                    except ValueError as exc:
                        s["unusable"] = str(exc)
                        error = error or str(exc)
                        continue
                if "value" not in s:
                    s["value"] = round(WEIGHTS["pairwise"] * s["pairwise"] + WEIGHTS["style"] * s["style"]
                                       - WEIGHTS["off_topic"] * s["off_topic"], 6)
                members.append(s["value"])
            if error:
                rejected.append(group)
                reasons[error] = reasons.get(error, 0) + 1
            else:
                values[start:start + g] = members
        groups = n // g
        self.totals["batches"] += 1
        self.totals["groups"] += groups
        self.totals["rejected_groups"] += len(rejected)
        judged = [s["pairwise"] for s in samples if s.get("pairwise") is not None and "keys" in s]
        metrics = rule_metrics(replies, character)
        metrics.update(groups=groups, rejected_groups=len(rejected), rejection_reasons=reasons,
                       judge_calls=len(calls), reused_requests=wanted - len(pending),
                       win_rate_vs_base=sum(judged) / len(judged) if judged else None,
                       off_topic=sum(bool(s.get("off_topic")) for s in samples),
                       varied_groups=sum(len({v for v in values[s:s + g]}) > 1
                                         for k, s in enumerate(range(0, n, g)) if k not in rejected))
        self.last_metrics = metrics
        dump(self.output / f"rewards_{self.totals['batches']:05d}.json",
             dict(metrics=metrics, rejected_groups=rejected, samples=samples, values=values))
        self.pending_rejected = rejected
        self.all_rejected_streak = self.all_rejected_streak + 1 if len(rejected) == groups else 0
        if self.all_rejected_streak >= self.max_all_rejected:
            raise RuntimeError(f"every group rejected in {self.all_rejected_streak} consecutive batches: {reasons}")
        return values
