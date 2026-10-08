"""Style reward: rule gates and the judge's P+R+N margin over frozen base replies.

Per sample (policy and base replies both lose an outer 「」 first):
  hard violation (out of character, length, catchphrase cap, impersonation,
  multi-line)                         -> -1, no judge call
  byte-identical to the base reply    -> 0, no judge call
  otherwise                           -> AB/BA judge calls against the base reply;
                                         reward = mean over the two orders of
                                         (policy P+R+N - base P+R+N) / 12, in [-1, 1]

grpo_style_04 (style-pairwise-v3: 0.7 verdict + 0.3 rule style score) was
hacked through the rule score: a fixed 嗯... / 哼， opener earned a deterministic
gain inside every group while the verdict was mostly ties. The rule style score,
the verdict (1 / 0.5 / 0, AB/BA disagreement a flagged tie), off_topic (both
orders scored the policy's R <= 1) and order disagreement are now logged only.

A group that contains any unusable sample (judge failure, parse error,
insufficient) is rejected as a whole: every member returns None,
so TRL's nansum gives the group all-zero rewards and zero advantages, and the
trainer additionally zeroes its completion_mask. A partial None would be summed
as 0 and silently corrupt the group's mean and std, so it is never returned.

Under DDP each rank scores only its own rows and TRL concatenates them in rank
order, so a group can straddle two ranks. Every rank therefore exchanges its
per-sample outcomes and decides rejection, counters and metrics on the whole
batch: a failure on any rank rejects the group on all of them, and the
all-rejected streak is the same everywhere. Judge calls stay local to the rank.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from characore.judge import DIMS, PROTOCOL, make_request
from characore.judge_runner import call_judge, identity
from characore.persona import CHARACTERS, OTHER
from characore.protocol import dump

LENGTH = (1, 80)  # Rei's in-character "不。" is one character; per-persona brevity is the soft length term
# Any one tone marker more than this many times, or this many marker hits in total, is stuffing.
MARKER_CAP, MARKER_TOTAL_CAP = 2, 4
OOC = ("作为AI", "作为一个AI", "作为人工智能", "人工智能", "语言模型", "AI助手", "我是AI",
       "扮演", "角色设定", "设定中", "这个角色", "台词：", "旁白")
MARGIN_SCALE = 12  # P+R+N spans 0..12, so the mean margin lies in [-1, 1], the same range as the hard penalty
REWARD_SPEC = dict(name="style-margin-v4", judge_protocol=PROTOCOL, length=LENGTH,
                   marker_cap=MARKER_CAP, marker_total_cap=MARKER_TOTAL_CAP, ooc=OOC,
                   reward="mean over AB/BA of (policy P+R+N - base P+R+N) / 12",
                   hard_penalty=-1.0, identical_to_base=0.0,
                   outer_quotes="an outer 「」 wrapping the whole reply is removed from policy and base replies",
                   logged_only="rule style score, verdict win rate, off_topic, AB/BA verdict disagreement",
                   ellipsis="'...' and '…' runs count as '……' for markers only",
                   judge_retries="one retry on call failure or parse error; every attempt recorded",
                   canon="with --canon, the character's original lines are judge evidence E4 (bound by hash)",
                   rejection="call failure, parse error or insufficient rejects the whole group; never zero-filled")
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")
ELLIPSIS = re.compile(r"\.{3,}|…+")
SELF_TITLE = "本小姐"
# grpo_style_04's learned openers (after marker_text); logged so a return of either shows on the curves.
OPENERS = {"rei": ("嗯", "……"), "asuka": ("哼",)}


def clean_reply(raw):
    """Strip the empty think block some templates emit, surrounding whitespace, and an outer 「」.

    The base sometimes wraps the whole line in 「」; that is formatting, not voice, and it made two
    otherwise identical replies differ. Inner quotes ("他说「走」") are kept.
    """
    reply = EMPTY_THINK.sub("", raw).strip()
    if len(reply) > 2 and reply[0] == "「" and reply[-1] == "」" and not any(q in reply[1:-1] for q in "「」"):
        reply = reply[1:-1].strip()
    return reply


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
    # The card says Asuka only occasionally calls herself 本小姐; the base opens 97% of her lines with it.
    asuka = [r for r, c in zip(replies, characters) if c == "asuka"]
    return dict(samples=n, hard_violation_rate=sum(h is not None for h in hard) / n if n else None,
                asuka_benxiaojie_open_rate=sum(r.startswith(SELF_TITLE) for r in asuka) / len(asuka) if asuka else None,
                stock_opener_rate=sum(marker_text(r).startswith(OPENERS[c]) for r, c in zip(replies, characters)) / n if n else None,
                hard_reasons=reasons,
                style_score_mean=sum(style_score(r, c) for r, c in zip(replies, characters)) / n if n else None,
                length_mean=sum(len(r) for r in replies) / n if n else None,
                marker_rate=sum(any(m in marker_text(r) for m in CHARACTERS[c]["markers"]) for r, c in zip(replies, characters)) / n if n else None)


def _original(final, mapping):
    """One order's verdict and per-dimension scores, translated back to original labels.

    Returns (winner, {"A": {dim: score}, "B": {dim: score}}); original A is the policy reply.
    Raises ValueError("insufficient") when the verdict or any dimension is insufficient.
    """
    judgement = final["judgement"]
    winner = judgement["winner"]
    if winner == "insufficient":
        raise ValueError("insufficient")
    scores = {}
    for side, orig in mapping.items():
        entry = judgement["scores"][side]
        if any(entry[d]["status"] != "scored" for d in DIMS):
            raise ValueError("insufficient")
        scores[orig] = {d: entry[d]["score"] for d in DIMS}
    return (mapping[winner] if winner in ("A", "B") else winner), scores


def pair_outcome(ab, ba):
    """Combine AB and BA results for policy (original A) against base (original B).

    Returns dict(margin, policy, base, verdict, off_topic, order_inconsistent):
      margin   mean over the two orders of (policy P+R+N - base P+R+N) / 12, in [-1, 1]; the reward
      policy, base  per-order P+R+N totals, AB first
      verdict  1 / 0.5 / 0 for policy win / tie / loss; AB/BA disagreement is a tie (logged only)
      off_topic  both orders scored the policy's R <= 1 (logged only)
    Raises ValueError with the reason when a call failed or anything was insufficient.
    """
    for label, call in (("AB", ab), ("BA", ba)):
        if call["final"]["call_status"] != "ok":
            raise ValueError(f"{label} {call['final']['call_status']}")
    first, s_first = _original(ab["final"], ab["mapping"])
    second, s_second = _original(ba["final"], ba["mapping"])
    policy = [sum(s["A"].values()) for s in (s_first, s_second)]
    base = [sum(s["B"].values()) for s in (s_first, s_second)]
    margin = round(sum(p - b for p, b in zip(policy, base)) / (2 * MARGIN_SCALE), 6)
    inconsistent = first != second
    verdict = 0.5 if inconsistent else {"A": 1.0, "tie": 0.5, "B": 0.0}[first]
    off_topic = s_first["A"]["R"] <= 1 and s_second["A"]["R"] <= 1
    return dict(margin=margin, policy=policy, base=base, verdict=verdict, off_topic=off_topic,
                order_inconsistent=inconsistent)


class PairJudge:
    """AB/BA judge calls with in-batch reuse of byte-identical requests."""

    # One retry on call failure or parse error: survey_style_02 lost 5 of 20 groups to single format
    # slips. Both attempts are recorded. Order disagreement is not a failure and is never retried.
    # canon: frozen reference lines (characore.canon.load_canon) sent as evidence E4, or None.
    def __init__(self, judge, output, workers=8, retries=1, resume=False, canon=None):
        self.judge, self.output, self.workers, self.retries = judge, Path(output), workers, retries
        self.canon = canon
        self.output.mkdir(parents=True, exist_ok=resume)
        # A resumed run continues numbering past every batch already on disk; nothing is overwritten.
        self.batches = max((int(p.name.split("_")[1]) for p in self.output.glob("batch_*")), default=0)

    def requests(self, row, reply, base):
        out = {}
        for reverse in (False, True):
            request = make_request(row, {"A": reply, "B": base}, reverse=reverse, canon=self.canon)
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


class SingleProcess:
    """Cross-rank exchange for one process: the whole batch is this rank's."""
    rank, world = 0, 1

    def gather(self, value):
        return [value]

    def barrier(self):
        pass


class TorchDistributed:
    """all_gather_object over the initialised default process group, one entry per rank in rank order."""

    def __init__(self):
        import torch.distributed as dist
        self.dist, self.rank, self.world = dist, dist.get_rank(), dist.get_world_size()

    def gather(self, value):
        out = [None] * self.world
        self.dist.all_gather_object(out, value)
        return out

    def barrier(self):
        self.dist.barrier()


class StyleReward:
    """TRL reward function. The global batch (all ranks' rows in rank order) holds whole consecutive groups of G."""

    def __init__(self, judge, output, group_size, workers=8, max_all_rejected=3, resume=False, sync=None,
                 canon=None):
        self.__name__ = "style"  # TRL names the reward column after this
        self.pairs = PairJudge(judge, Path(output) / "calls", workers, resume=resume, canon=canon)
        self.output = Path(output)
        self.sync = sync or SingleProcess()
        self.group_size = group_size
        self.max_all_rejected = max_all_rejected
        self.all_rejected_streak = 0
        self.pending_rejected = None
        self.last_metrics = None
        # totals cover the whole batch and are identical on every rank; local_judge_calls is this rank's own.
        self.totals = dict(batches=0, groups=0, rejected_groups=0, judge_calls=0, reused_requests=0,
                           skipped_identical=0, hard=0, order_inconsistent=0)
        self.local_judge_calls = 0
        # rewards_{serial}.json numbering; equals totals["batches"] unless the run was resumed.
        self.serial = max((int(p.stem.split("_")[1]) for p in self.output.glob("rewards_*.json")), default=0)

    def state(self):
        """Counters a checkpoint carries so that a resumed run continues them."""
        return dict(totals=dict(self.totals), all_rejected_streak=self.all_rejected_streak, serial=self.serial,
                    local_judge_calls=self.local_judge_calls)

    def restore(self, state):
        """Continue from a checkpoint's counters (None: no checkpoint, start from zero). Batches scored
        after that checkpoint were never part of a saved update; their files stay on disk and are
        returned as superseded."""
        state = state or dict(totals={k: 0 for k in self.totals}, all_rejected_streak=0, serial=0,
                              local_judge_calls=0)
        self.totals = dict(state["totals"])
        self.all_rejected_streak = state["all_rejected_streak"]
        self.local_judge_calls = state["local_judge_calls"]
        return list(range(state["serial"] + 1, self.serial + 1))

    def take_rejected(self):
        """Global group indices rejected in the latest call. The trainer must consume exactly one per batch."""
        if self.pending_rejected is None:
            raise RuntimeError("reward was not computed for this batch")
        rejected, self.pending_rejected = self.pending_rejected, None
        return rejected

    def __call__(self, prompts, completions, completion_ids=None, row_id=None, character=None,
                 base_reply=None, situation=None, speaker=None, line=None, **kwargs):
        g, sync = self.group_size, self.sync
        n = len(completions)
        if not (len(prompts) == len(row_id) == len(character) == len(base_reply) == n):
            raise ValueError("reward batch columns differ in length")
        # Check the global grouping before any judge call. Every rank raises together, so none is left waiting.
        layout = sync.gather(dict(rows=n, keys=[(r, p) for r, p in zip(row_id, prompts)]))
        if any(e["rows"] != n for e in layout):
            raise ValueError("ranks scored different numbers of rows")
        keys = [k for e in layout for k in e["keys"]]
        total, offset = n * sync.world, n * sync.rank
        if total % g:
            raise ValueError("reward batch is not whole groups")
        for start in range(0, total, g):
            if len(set(keys[start:start + g])) != 1:
                raise ValueError("group members must share one prompt; ordering assumption broken")
        replies = [clean_reply(c) for c in completions]
        # Base replies frozen before v4 may still carry an outer 「」; both sides are cleaned the same way.
        bases = [clean_reply(b) for b in base_reply]
        rows = [dict(character=character[i], situation=situation[i], speaker=speaker[i], line=line[i])
                for i in range(n)]
        samples, pending, wanted = [], {}, 0
        for i, reply in enumerate(replies):
            sample = dict(row_id=row_id[i], reply=reply, hard=hard_violation(reply, character[i]),
                          style=style_score(reply, character[i]))
            if sample["hard"] is None and reply != bases[i]:
                reqs = self.pairs.requests(rows[i], reply, bases[i])
                sample["keys"] = {label: key for label, (key, _) in reqs.items()}
                for key, request in reqs.values():
                    wanted += 1
                    pending.setdefault(key, request)
            samples.append(sample)
        calls = self.pairs.run(pending) if pending else {}
        self.local_judge_calls += len(calls)
        # Each sample's outcome needs only its own calls; whether it is usable is decided per group below.
        for s in samples:
            if s["hard"]:
                s.update(off_topic=False, value=REWARD_SPEC["hard_penalty"])
            elif "keys" not in s:
                s.update(off_topic=False, identical_to_base=True, value=REWARD_SPEC["identical_to_base"])
            else:
                try:
                    s.update(pair_outcome(calls[s["keys"]["AB"]], calls[s["keys"]["BA"]]))
                except ValueError as exc:
                    s["unusable"] = str(exc)
                    continue
                s["value"] = s["margin"]

        # Every rank sees the same global batch from here on, so every decision below is identical across ranks.
        everyone = sync.gather(dict(samples=samples, replies=replies, characters=list(character),
                                    calls=len(calls), reused=wanted - len(pending)))
        merged = lambda key: [x for e in everyone for x in e[key]]
        all_samples = merged("samples")
        batch_calls, batch_reused = sum(e["calls"] for e in everyone), sum(e["reused"] for e in everyone)
        self.totals["judge_calls"] += batch_calls
        self.totals["reused_requests"] += batch_reused

        values, rejected, reasons = [None] * total, [], {}
        for group, start in enumerate(range(0, total, g)):
            members = all_samples[start:start + g]
            self.totals["hard"] += sum(bool(s["hard"]) for s in members)
            self.totals["skipped_identical"] += sum(bool(s.get("identical_to_base")) for s in members)
            self.totals["order_inconsistent"] += sum(bool(s.get("order_inconsistent")) for s in members)
            error = next((s["unusable"] for s in members if "unusable" in s), None)
            if error:
                rejected.append(group)
                reasons[error] = reasons.get(error, 0) + 1
            else:
                values[start:start + g] = [s["value"] for s in members]
        groups = total // g
        self.totals["batches"] += 1
        self.totals["groups"] += groups
        self.totals["rejected_groups"] += len(rejected)
        compared = [s for s in all_samples if "margin" in s]
        judged = [s["verdict"] for s in compared]
        margins = [s["margin"] for s in compared]
        inconsistent = sum(s["order_inconsistent"] for s in compared)
        usable = [s["value"] for s in all_samples if "value" in s]
        metrics = rule_metrics(merged("replies"), merged("characters"))
        metrics.update(groups=groups, rejected_groups=len(rejected), rejection_reasons=reasons,
                       judge_calls=batch_calls, reused_requests=batch_reused,
                       reward_mean=sum(usable) / len(usable) if usable else None,
                       margin_mean=sum(margins) / len(margins) if margins else None,
                       margin_positive_rate=sum(m > 0 for m in margins) / len(margins) if margins else None,
                       policy_prn_mean=sum(sum(s["policy"]) / 2 for s in compared) / len(compared) if compared else None,
                       base_prn_mean=sum(sum(s["base"]) / 2 for s in compared) / len(compared) if compared else None,
                       identical_rate=sum(bool(s.get("identical_to_base")) for s in all_samples) / total,
                       win_rate_vs_base=sum(judged) / len(judged) if judged else None,
                       order_inconsistent=inconsistent,
                       order_inconsistent_rate=inconsistent / len(compared) if compared else None,
                       off_topic=sum(bool(s.get("off_topic")) for s in all_samples),
                       varied_groups=sum(len({v for v in values[s:s + g]}) > 1
                                         for k, s in enumerate(range(0, total, g)) if k not in rejected))
        self.last_metrics = metrics
        self.serial += 1
        record = dict(metrics=metrics, rejected_groups=rejected, samples=samples, values=values[offset:offset + n])
        if sync.world > 1:
            # This rank's rows only; rejected_groups and metrics cover the whole batch.
            record.update(rank=sync.rank, row_offset=offset, local_judge_calls=len(calls))
        dump(self.output / f"rewards_{self.serial:05d}.json", record)
        self.pending_rejected = rejected
        self.all_rejected_streak = self.all_rejected_streak + 1 if len(rejected) == groups else 0
        if self.all_rejected_streak >= self.max_all_rejected:
            raise RuntimeError(f"every group rejected in {self.all_rejected_streak} consecutive batches: {reasons}")
        return values[offset:offset + n]
