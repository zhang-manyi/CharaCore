"""Style reward, judge protocol, suite freeze and group-rejection checks. Standard library only
(apply_rejection uses torch when it is importable)."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from characore.canon import canon_identity, load_canon
from characore.judge import (ATTR_PROTOCOL, PROTOCOL, make_attribution_request, make_request,
                             parse_attribution, parse_response)
from characore.persona import CHARACTERS, persona_identity, policy_messages
from characore.protocol import digest, dump
from characore.stub_judge import StubJudge
from characore.style_data import load_base_replies, load_suite, select_character
from characore.style_rewards import (StyleReward, clean_reply, hard_violation, pair_outcome,
                                     rule_metrics, style_score)
from characore.style_trainer import apply_rejection

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "experiments/style_v2"  # style_v1 rows rebound to the v4 cards
CANON = ROOT / "experiments/canon_v1"
ROW = dict(id="T01-L1-rei", template="T01", character="rei", situation="放学后的教室。", speaker="同班同学",
           line="你带伞了吗？")
IDS = ["E1", "E2", "E3", "candidate:A", "candidate:B"]


def judgement(winner="A", r=3, protocol=PROTOCOL):
    """The displayed winner scores P=4 and the loser P=2 (a tie: both 3), so the P+R+N margin is +-2 per order."""
    dim = lambda score: dict(status="scored", score=score, reason="ok", evidence_ids=["E1"])
    p = lambda s: 3 if winner not in ("A", "B") else (4 if s == winner else 2)
    side = lambda s: {"P": dim(p(s)), "R": dim(r), "N": dim(3)}
    return dict(protocol=protocol, winner=winner, scores={"A": side("A"), "B": side("B")}, reason="ok",
                preference_evidence_ids=[] if winner in ("tie", "insufficient") else ["E1"])


WIN = round(4 / 24, 6)  # policy +2 in both orders, over 2 orders x 12


def call(winner, mapping, r=3, status="ok"):
    final = dict(call_status=status, judgement=judgement(winner, r) if status == "ok" else None)
    return dict(final=final, mapping=mapping)


class ScriptedJudge:
    """Returns a fixed winner per displayed order, so order disagreement can be forced."""

    def __init__(self, ab="A", ba="B", fail_on=None):
        self.ab, self.ba, self.fail_on, self.calls = ab, ba, fail_on, 0
        self.metadata = dict(kind="stub_judge", claim="test")
        self.budget = {}

    def check_input(self, messages, input_limit=None):
        return None

    def __call__(self, messages):
        self.calls += 1
        data = json.loads(messages[1]["content"])
        if self.fail_on and self.fail_on in data["candidate_A"] + data["candidate_B"]:
            return "not json", dict(finish_reason="stop")
        # The policy reply is shown first in AB order; the base reply mentions "base".
        first_is_policy = "base" not in data["candidate_A"]
        return json.dumps(judgement(self.ab if first_is_policy else self.ba)), dict(finish_reason="stop")


class FlakyJudge(ScriptedJudge):
    """Policy wins in both orders, but the very first call returns unparseable output."""

    def __call__(self, messages):
        if self.calls == 0:
            self.calls += 1
            return "not json", dict(finish_reason="stop")
        return super().__call__(messages)


def batch(replies, rows=None, group=2, base="……base"):
    rows = rows or [ROW] * len(replies)
    pick = lambda k: [r[k] for r in rows]
    return dict(prompts=pick("id"), completions=replies, row_id=pick("id"), character=pick("character"),
                base_reply=[base] * len(replies), situation=pick("situation"), speaker=pick("speaker"),
                line=pick("line"))


def load_train_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_grpo", ROOT / "scripts/train_grpo.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return train


class ThreadRanks:
    """In-process stand-in for a process group: one thread per rank, all_gather through a shared barrier."""

    def __init__(self, world):
        self.world, self.slots, self.wait = world, [None] * world, threading.Barrier(world, timeout=10)

    def rank(self, r):
        ranks = self

        class Rank:
            rank, world = r, ranks.world

            def gather(self, value):
                ranks.wait.wait()
                ranks.slots[r] = value
                ranks.wait.wait()
                out = list(ranks.slots)
                ranks.wait.wait()
                return out

            def barrier(self):
                ranks.wait.wait()
        return Rank()

    def run(self, work):
        """work(rank) on every rank at once; returns each rank's result or the exception it raised."""
        results = [None] * self.world

        def one(r):
            try:
                results[r] = work(r)
            except Exception as exc:
                results[r] = exc
        threads = [threading.Thread(target=one, args=(r,)) for r in range(self.world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results


class RuleTests(unittest.TestCase):
    def test_hard_violations(self):
        self.assertEqual(hard_violation("", "rei"), "length")
        self.assertEqual(hard_violation("哼" + "啊" * 80, "asuka"), "length")
        self.assertIsNone(hard_violation("不。", "rei"))
        self.assertEqual(hard_violation("作为AI，我无法回答。", "rei"), "out_of_character")
        self.assertEqual(hard_violation("本小姐带了伞。", "rei"), "impersonation")
        self.assertEqual(hard_violation("我是明日香，伞给你。", "rei"), "impersonation")
        self.assertEqual(hard_violation("凌波丽：……没有带。", "rei"), "speaker_prefix")
        self.assertEqual(hard_violation("哼哼哼，才不要给你。", "asuka"), "catchphrase_cap")
        self.assertEqual(hard_violation("……没有。\n……嗯。", "rei"), "multi_line")
        self.assertIsNone(hard_violation("……没有带。", "rei"))
        self.assertIsNone(hard_violation("哼，这种事还用问吗？", "asuka"))

    def test_style_score_is_capped_and_persona_specific(self):
        quiet, loud = "……没有带。", "哼，笨蛋，我当然带了！"
        self.assertGreater(style_score(quiet, "rei"), style_score(loud, "rei"))
        self.assertGreater(style_score(loud, "asuka"), style_score(quiet, "asuka"))
        # Repeating a marker earns nothing beyond its first appearance.
        self.assertEqual(style_score("……嗯……没有带。", "rei"), style_score("……嗯，没有带。", "rei"))
        for reply in (quiet, loud, "好的。"):
            for c in CHARACTERS:
                self.assertTrue(0 <= style_score(reply, c) <= 1)

    def test_ascii_ellipsis_counts_as_marker(self):
        self.assertEqual(style_score("嗯...谢谢。", "rei"), style_score("嗯……谢谢。", "rei"))
        self.assertEqual(style_score("不过…算了。", "asuka"), style_score("不过……算了。", "asuka"))
        self.assertEqual(hard_violation("...嗯...嗯...", "rei"), "catchphrase_cap")
        self.assertEqual(clean_reply("嗯..."), "嗯...")  # text itself is never rewritten

    def test_clean_reply_strips_empty_think_and_outer_quotes(self):
        self.assertEqual(clean_reply("<think>\n\n</think>\n\n……没有。 "), "……没有。")
        self.assertEqual(clean_reply("<think>想</think>没有"), "<think>想</think>没有")
        self.assertEqual(clean_reply("「没有带。」"), "没有带。")
        self.assertEqual(clean_reply("<think></think>「 没有带。」"), "没有带。")
        # Only a wrapper around the whole reply goes; inner or partial quotes stay.
        self.assertEqual(clean_reply("他说「走」。"), "他说「走」。")
        self.assertEqual(clean_reply("「走」「留」"), "「走」「留」")
        self.assertEqual(clean_reply("「」"), "「」")

    def test_outer_quotes_do_not_make_replies_differ(self):
        reward = StyleRewardTests.reward(self, ScriptedJudge())
        values = reward(**batch(["「……base」", "……base"], base="「……base」"))
        self.assertEqual(values, [0.0, 0.0])
        self.assertEqual(reward.totals["skipped_identical"], 2)

    def test_rule_metrics_counts(self):
        m = rule_metrics(["……没有带。", "作为AI我不知道"], ["rei", "rei"])
        self.assertEqual(m["hard_violation_rate"], 0.5)
        self.assertEqual(m["hard_reasons"], {"out_of_character": 1})
        self.assertIsNone(m["asuka_benxiaojie_open_rate"])

    def test_benxiaojie_is_logged_not_scored(self):
        # Same length, so only the self-title differs.
        self.assertEqual(style_score("本小姐当然带了伞。", "asuka"), style_score("我今天当然带了伞。", "asuka"))
        m = rule_metrics(["本小姐带了。", "哼，带了。", "……嗯。"], ["asuka", "asuka", "rei"])
        self.assertEqual(m["asuka_benxiaojie_open_rate"], 0.5)


class JudgeProtocolTests(unittest.TestCase):
    def test_request_hides_mapping_and_swaps(self):
        ab = make_request(ROW, {"A": "x", "B": "y"})
        ba = make_request(ROW, {"A": "x", "B": "y"}, reverse=True)
        data = json.loads(ba["messages"][1]["content"])
        self.assertEqual((data["candidate_A"], data["candidate_B"]), ("y", "x"))
        self.assertNotIn("display_to_original", ba["messages"][1]["content"])
        self.assertEqual(ab["allowed_evidence_ids"], IDS)
        self.assertNotIn("original", ab["messages"][1]["content"])

    def test_strict_parse(self):
        self.assertEqual(parse_response(json.dumps(judgement()), IDS)["call_status"], "ok")
        bad = []
        j = judgement(); j["extra"] = 1; bad.append(j)
        j = judgement(); j["preference_evidence_ids"] = []; bad.append(j)
        j = judgement(); j["scores"]["A"]["P"]["score"] = 5; bad.append(j)
        j = judgement(); j["scores"]["A"]["P"]["score"] = 3.0; bad.append(j)
        j = judgement(); j["scores"]["A"]["P"]["evidence_ids"] = ["E9"]; bad.append(j)
        j = judgement(); del j["scores"]["B"]["N"]; bad.append(j)
        j = judgement(); j["scores"]["A"]["P"]["status"] = "not_applicable"; bad.append(j)
        j = judgement(); j["protocol"] = "judge-v0.3"; bad.append(j)
        for item in bad:
            self.assertEqual(parse_response(json.dumps(item), IDS)["call_status"], "parse_error", item)
        duplicate = json.dumps(judgement())[:-1] + ', "reason": "again"}'
        self.assertEqual(parse_response(duplicate, IDS)["call_status"], "parse_error")
        self.assertEqual(parse_response("```json\n{}\n```", IDS)["call_status"], "parse_error")
        tie = judgement("tie")
        self.assertEqual(parse_response(json.dumps(tie), IDS)["call_status"], "ok")
        # survey_style_02: a dropped brace nested reason and preference_evidence_ids inside scores.
        j = judgement(); j["scores"]["reason"] = j.pop("reason")
        j["scores"]["preference_evidence_ids"] = j.pop("preference_evidence_ids")
        self.assertEqual(parse_response(json.dumps(j), IDS)["error"], "Unexpected judge fields")

    def test_prompt_asks_for_analysis_before_winner_and_scores_last(self):
        system = make_request(ROW, {"A": "……嗯。", "B": "……不。"})["messages"][0]["content"]
        self.assertIn("protocol, reason, winner, preference_evidence_ids, scores", system)
        self.assertIn(f'protocol 填 "{PROTOCOL}"', system)

    def test_attribution_protocol(self):
        req = make_attribution_request(ROW, "……没有。", reverse=True)
        self.assertEqual(req["card_order"], ["asuka", "rei"])
        self.assertNotIn('"character"', req["messages"][1]["content"])
        ok = dict(protocol=ATTR_PROTOCOL, speaker="rei", reason="平淡")
        self.assertEqual(parse_attribution(json.dumps(ok))["call_status"], "ok")
        for bad in (dict(ok, speaker="shinji"), dict(ok, extra=1), dict(ok, reason=" ")):
            self.assertEqual(parse_attribution(json.dumps(bad))["call_status"], "parse_error")

    def test_policy_prompt_carries_only_card_situation_line(self):
        text = json.dumps(policy_messages(ROW), ensure_ascii=False)
        self.assertIn(CHARACTERS["rei"]["card"], text)
        self.assertIn(ROW["line"], text)
        self.assertNotIn(CHARACTERS["asuka"]["card"], text)
        self.assertNotIn(ROW["id"], text)
        canon = load_canon(CANON)
        self.assertFalse(any(e["line"] in text for entries in canon.values() for e in entries))

    def test_canon_is_judge_evidence_e4_only_with_canon(self):
        canon = load_canon(CANON)
        plain = make_request(ROW, {"A": "x", "B": "y"})
        with_canon = make_request(ROW, {"A": "x", "B": "y"}, canon=canon)
        data = json.loads(with_canon["messages"][1]["content"])
        self.assertEqual([e["id"] for e in data["evidence"]], ["E1", "E2", "E3", "E4"])
        self.assertIn(canon["rei"][0]["line"], data["evidence"][3]["text"])
        self.assertNotIn(canon["asuka"][0]["line"], data["evidence"][3]["text"])  # own character only
        self.assertEqual(with_canon["allowed_evidence_ids"], ["E1", "E2", "E3", "E4", "candidate:A", "candidate:B"])
        self.assertEqual(plain["allowed_evidence_ids"], IDS)
        system = with_canon["messages"][0]["content"]
        self.assertIn("E4 原作参考台词", system)
        self.assertNotIn("你对原作", system)  # memory is replaced by the shown lines
        self.assertIn("你对原作", plain["messages"][0]["content"])
        reply = dict(judgement(), preference_evidence_ids=["E4"])
        self.assertEqual(parse_response(json.dumps(reply), with_canon["allowed_evidence_ids"])["call_status"], "ok")
        self.assertEqual(parse_response(json.dumps(reply), plain["allowed_evidence_ids"])["call_status"], "parse_error")

    def test_canon_freeze_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "canon"
            copy.mkdir()
            for name in ("canon.json", "freeze.json"):
                (copy / name).write_bytes((CANON / name).read_bytes())
            self.assertEqual(set(load_canon(copy)), {"rei", "asuka"})
            (copy / "canon.json").write_bytes((CANON / "canon.json").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_canon(copy)
        self.assertIsNone(canon_identity(None))
        self.assertEqual(canon_identity(CANON), digest(CANON / "canon.json"))


class PairOutcomeTests(unittest.TestCase):
    AB, BA = {"A": "A", "B": "B"}, {"A": "B", "B": "A"}

    def outcome(self, ab, ba):
        o = pair_outcome(ab, ba)
        return o["margin"], o["verdict"], o["order_inconsistent"]

    def test_mirrored_winners_agree(self):
        self.assertEqual(self.outcome(call("A", self.AB), call("B", self.BA)), (WIN, 1.0, False))
        self.assertEqual(self.outcome(call("B", self.AB), call("A", self.BA)), (-WIN, 0.0, False))
        self.assertEqual(self.outcome(call("tie", self.AB), call("tie", self.BA)), (0.0, 0.5, False))
        o = pair_outcome(call("A", self.AB), call("B", self.BA))
        self.assertEqual((o["policy"], o["base"]), ([10, 10], [8, 8]))

    def test_margin_maps_scores_back_through_the_order(self):
        # BA shows the base reply first: displayed A's scores belong to the base.
        ab = call("tie", self.AB)
        ba = call("tie", self.BA)
        ba["final"]["judgement"]["scores"]["A"]["N"]["score"] = 0  # base, shown first, scores N=0 in BA
        o = pair_outcome(ab, ba)
        self.assertEqual((o["policy"], o["base"], o["margin"]), ([9, 9], [9, 6], round(3 / 24, 6)))

    def test_order_disagreement_is_logged_and_margins_cancel(self):
        # Position-following (A, A): +2 in AB and -2 in BA average to 0; the verdict is a flagged tie.
        self.assertEqual(self.outcome(call("A", self.AB), call("A", self.BA)), (0.0, 0.5, True))
        self.assertEqual(self.outcome(call("tie", self.AB), call("B", self.BA)), (round(2 / 24, 6), 0.5, True))

    def test_failure_and_insufficient_are_rejected(self):
        partial = call("A", self.AB)
        partial["final"]["judgement"]["scores"]["B"]["R"].update(status="insufficient", score=None)
        for ab, ba, reason in ((call("A", self.AB, status="parse_error"), call("B", self.BA), "AB parse_error"),
                               (call("insufficient", self.AB), call("insufficient", self.BA), "insufficient"),
                               (partial, call("B", self.BA), "insufficient")):
            with self.assertRaisesRegex(ValueError, reason):
                pair_outcome(ab, ba)

    def test_off_topic_needs_both_orders(self):
        self.assertTrue(pair_outcome(call("B", self.AB, r=1), call("A", self.BA, r=0))["off_topic"])
        self.assertFalse(pair_outcome(call("B", self.AB, r=1), call("A", self.BA, r=3))["off_topic"])


class StyleRewardTests(unittest.TestCase):
    def reward(self, judge, group=2):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return StyleReward(judge, Path(tmp.name) / "r", group)

    def test_usable_batch_values_and_reuse(self):
        judge = ScriptedJudge("A", "B")  # policy wins in both orders
        reward = self.reward(judge)
        values = reward(**batch(["……没有带。", "……没有带。", "作为AI我不知道", "……base"]))
        # The rule style score no longer enters the value: margin, hard -1, identical 0.
        self.assertEqual(values, [WIN, WIN, -1.0, 0.0])
        self.assertEqual(reward.take_rejected(), [])
        self.assertEqual(reward.last_metrics["margin_mean"], WIN)
        self.assertEqual(reward.last_metrics["win_rate_vs_base"], 1.0)
        self.assertEqual(reward.last_metrics["identical_rate"], 0.25)
        # Two identical policy replies share one AB and one BA call.
        self.assertEqual(judge.calls, 2)
        self.assertEqual(reward.last_metrics["reused_requests"], 2)
        self.assertEqual(reward.totals["skipped_identical"], 1)
        with self.assertRaises(RuntimeError):
            reward.take_rejected()

    def test_canon_reaches_every_judge_request(self):
        seen = []

        class Recording(ScriptedJudge):
            def __call__(self, messages):
                seen.append([e["id"] for e in json.loads(messages[1]["content"])["evidence"]])
                return super().__call__(messages)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        reward = StyleReward(Recording("A", "B"), Path(tmp.name) / "r", 2, canon=load_canon(CANON))
        self.assertEqual(reward(**batch(["……没有带。", "……base"])), [WIN, 0.0])
        self.assertEqual(seen, [["E1", "E2", "E3", "E4"]] * 2)

    def test_order_disagreement_scores_tie_and_is_counted(self):
        judge = ScriptedJudge("A", "A")  # winner follows display position: inconsistent
        reward = self.reward(judge)
        values = reward(**batch(["……没有带。", "作为AI我不知道"]))
        self.assertEqual(values, [0.0, -1.0])  # +2 then -2: position bias cancels in the margin
        self.assertEqual(reward.take_rejected(), [])
        self.assertEqual(reward.last_metrics["order_inconsistent"], 1)
        self.assertEqual(reward.last_metrics["order_inconsistent_rate"], 1.0)
        self.assertEqual(reward.totals["order_inconsistent"], 1)

    def test_insufficient_rejects_whole_group_never_zero(self):
        reward = self.reward(ScriptedJudge("insufficient", "insufficient"))
        values = reward(**batch(["……没有带。", "作为AI我不知道", "……base", "……base"]))
        self.assertEqual(values[:2], [None, None])
        self.assertEqual(reward.take_rejected(), [0])
        self.assertEqual(reward.last_metrics["rejection_reasons"], {"insufficient": 1})

    def test_parse_failure_rejects_group_and_streak_aborts(self):
        reward = self.reward(ScriptedJudge(fail_on="不知道"))
        for _ in range(2):
            reward(**batch(["……不知道。", "……也不知道。"]))
            self.assertEqual(reward.take_rejected(), [0])
        with self.assertRaisesRegex(RuntimeError, "consecutive"):
            reward(**batch(["……不知道。", "……还是不知道。"]))

    def test_single_parse_failure_is_retried_once(self):
        judge = FlakyJudge()
        reward = self.reward(judge)
        values = reward(**batch(["……没有带。", "作为AI我不知道"]))
        self.assertEqual(reward.take_rejected(), [])
        self.assertEqual(values[0], WIN)
        self.assertEqual(judge.calls, 3)  # AB, BA, and one retry of the first call

    def test_batch_must_be_whole_consistent_groups(self):
        reward = self.reward(ScriptedJudge())
        with self.assertRaisesRegex(ValueError, "whole groups"):
            reward(**batch(["……没有带。"] * 3))
        other = dict(ROW, id="T01-L2-rei")
        with self.assertRaisesRegex(ValueError, "share one prompt"):
            reward(**batch(["……没有带。", "……没有。"], rows=[ROW, other]))

    def test_resume_continues_numbering_and_restores_counters(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "r"
        first = StyleReward(ScriptedJudge("A", "B"), out, 2)
        first(**batch(["……没有带。", "……嗯。"]))
        saved = json.loads(json.dumps(first.state()))  # checkpoint after batch 1
        first(**batch(["……不。", "……是吗。"]))  # batch 2: scored, then the process dies
        with self.assertRaises(FileExistsError):
            StyleReward(ScriptedJudge(), out, 2)  # a fresh run never reuses a directory
        resumed = StyleReward(ScriptedJudge("A", "B"), out, 2, resume=True)
        self.assertEqual(resumed.restore(saved), [2])  # batch 2 is superseded, its files kept
        self.assertEqual(resumed.totals, saved["totals"])
        resumed(**batch(["……不。", "……是吗。"]))
        self.assertEqual(resumed.totals["batches"], 2)
        self.assertEqual(sorted(p.name for p in out.glob("rewards_*.json")),
                         ["rewards_00001.json", "rewards_00002.json", "rewards_00003.json"])
        self.assertEqual(sorted(p.name for p in (out / "calls").glob("batch_*")),
                         ["batch_00001", "batch_00002", "batch_00003"])
        self.assertEqual(StyleReward(ScriptedJudge(), Path(tmp.name) / "s", 2).restore(None), [])

    def test_checkpoint_pickles_must_match_recorded_hashes(self):
        train = load_train_script()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = Path(tmp.name) / "checkpoint-5"
        ckpt.mkdir()
        (ckpt / "optimizer.pt").write_bytes(b"saved by this run")
        (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
        saved = dict(files_sha256=train.checkpoint_hashes(ckpt))
        train.verify_checkpoint(ckpt, saved)
        with self.assertRaisesRegex(ValueError, "no recorded file hashes"):
            train.verify_checkpoint(ckpt, {})
        (ckpt / "rng_state.pth").write_bytes(b"dropped in later")
        with self.assertRaisesRegex(ValueError, r"unrecorded \['rng_state.pth'\]"):
            train.verify_checkpoint(ckpt, saved)
        (ckpt / "rng_state.pth").unlink()
        (ckpt / "optimizer.pt").write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, r"changed \['optimizer.pt'\]"):
            train.verify_checkpoint(ckpt, saved)
        self.assertEqual(train.last_checkpoint(tmp.name), None)  # no reward state: incomplete save

    def test_checkpoint_needs_every_rank_state(self):
        train = load_train_script()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = Path(tmp.name) / "checkpoint-5"
        ckpt.mkdir()
        (ckpt / "rng_state_1.pth").write_bytes(b"rank 1 rng")
        (ckpt / "style_reward_rank1.json").write_text("{}")
        self.assertEqual(train.state_files(2), ["style_reward.json", "style_reward_rank1.json"])
        self.assertEqual(list(train.checkpoint_hashes(ckpt)), ["rng_state_1.pth"])  # state files record, not hashed
        self.assertIsNone(train.last_checkpoint(tmp.name, 2))  # rank 0 had not written its state yet
        (ckpt / "style_reward.json").write_text("{}")
        self.assertEqual(train.last_checkpoint(tmp.name, 2), ckpt)
        self.assertEqual(train.last_checkpoint(tmp.name), ckpt)
        self.assertEqual(train.comparable({}, "world_size"), 1)  # older manifests were one process

    def test_stub_judge_is_order_consistent(self):
        reward = self.reward(StubJudge())
        values = reward(**batch(["……没有带。", "哼，笨蛋，我当然带了！"], base="好的，我带了伞，可以借给你。"))
        self.assertEqual(reward.take_rejected(), [])
        self.assertGreater(values[0], values[1])


class TwoRankRewardTests(unittest.TestCase):
    """Group 4 over two ranks of 6 rows: group 1 is rank 0's rows 4-5 plus rank 1's rows 0-1."""
    RANK_REPLIES = (["……没有带。"] * 4 + ["……嗯。", "……是吗。"],
                    ["……不知道。", "……不。", "……好。", "……好。", "……走吧。", "……走吧。"])

    def rewards(self, ranks, judge=lambda: ScriptedJudge("A", "B", fail_on="不知道")):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return [StyleReward(judge(), Path(tmp.name) / f"rank{r}", 4, sync=ranks.rank(r)) for r in range(2)]

    def test_failure_on_one_rank_rejects_the_straddling_group_on_both(self):
        ranks = ThreadRanks(2)
        rewards = self.rewards(ranks)
        values = ranks.run(lambda r: rewards[r](**batch(self.RANK_REPLIES[r])))
        self.assertEqual(values[0][4:], [None, None])  # rank 0's judge saw no failure
        self.assertEqual(values[1][:2], [None, None])
        self.assertTrue(all(v is not None for v in values[0][:4] + values[1][2:]))
        self.assertEqual([r.take_rejected() for r in rewards], [[1], [1]])
        self.assertEqual(rewards[0].totals, rewards[1].totals)
        self.assertEqual(rewards[0].totals["groups"], 3)
        self.assertEqual(rewards[0].last_metrics["rejection_reasons"], rewards[1].last_metrics["rejection_reasons"])
        # Judge calls are local; the totals count both ranks'.
        local = [r.local_judge_calls for r in rewards]
        self.assertEqual(rewards[0].totals["judge_calls"], sum(local))
        record = json.loads((rewards[1].output / "rewards_00001.json").read_text(encoding="utf-8"))
        self.assertEqual((record["rank"], record["row_offset"], record["rejected_groups"]), (1, 6, [1]))
        self.assertEqual(len(record["samples"]), 6)

    def test_all_rejected_streak_aborts_both_ranks(self):
        ranks = ThreadRanks(2)
        rewards = self.rewards(ranks, judge=lambda: ScriptedJudge(fail_on="……"))
        for _ in range(2):
            ranks.run(lambda r: rewards[r](**batch(self.RANK_REPLIES[r])))
            self.assertEqual([r.take_rejected() for r in rewards], [[0, 1, 2], [0, 1, 2]])
        results = ranks.run(lambda r: rewards[r](**batch(self.RANK_REPLIES[r])))
        self.assertTrue(all(isinstance(e, RuntimeError) and "consecutive" in str(e) for e in results))

    def test_layout_mismatch_raises_on_both_before_any_call(self):
        ranks = ThreadRanks(2)
        judges = [ScriptedJudge(), ScriptedJudge()]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rewards = [StyleReward(judges[r], Path(tmp.name) / f"rank{r}", 4, sync=ranks.rank(r)) for r in range(2)]
        other = dict(ROW, id="T01-L2-rei")
        inputs = [batch(["……嗯。"] * 6), batch(["……嗯。"] * 6, rows=[ROW] + [other] * 5)]
        results = ranks.run(lambda r: rewards[r](**inputs[r]))
        self.assertTrue(all(isinstance(e, ValueError) and "share one prompt" in str(e) for e in results))
        self.assertEqual([j.calls for j in judges], [0, 0])
        results = ranks.run(lambda r: rewards[r](**batch(["……嗯。"] * (6 if r == 0 else 2))))
        self.assertTrue(all(isinstance(e, ValueError) and "different numbers" in str(e) for e in results))

    def test_each_rank_resumes_its_own_state(self):
        ranks = ThreadRanks(2)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        make = lambda r, resume=False: StyleReward(ScriptedJudge("A", "B", fail_on="不知道"), Path(tmp.name) / f"rank{r}",
                                                   4, resume=resume, sync=ranks.rank(r))
        first = [make(r) for r in range(2)]
        ranks.run(lambda r: first[r](**batch(self.RANK_REPLIES[r])))
        saved = [json.loads(json.dumps(f.state())) for f in first]
        self.assertNotEqual(saved[0]["local_judge_calls"], saved[1]["local_judge_calls"])
        ranks.run(lambda r: first[r](**batch(self.RANK_REPLIES[r])))  # scored after the checkpoint
        resumed = [make(r, resume=True) for r in range(2)]
        self.assertEqual([resumed[r].restore(saved[r]) for r in range(2)], [[2], [2]])
        self.assertEqual([r.local_judge_calls for r in resumed], [s["local_judge_calls"] for s in saved])
        ranks.run(lambda r: resumed[r](**batch(self.RANK_REPLIES[r])))
        self.assertEqual(resumed[0].totals, resumed[1].totals)
        self.assertEqual(resumed[0].totals["batches"], 2)


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch not installed")
class RejectionMaskTests(unittest.TestCase):
    def test_each_rank_masks_its_part_of_a_straddling_group(self):
        import torch
        masked = []
        for rank in range(2):  # 12 rows, group 4, rank 0 holds rows 0-5 and rank 1 rows 6-11
            output = dict(completion_mask=torch.ones(6, 2, dtype=torch.long), advantages=torch.ones(6))
            apply_rejection(output, [1], 4, offset=6 * rank, total=12)
            masked.append(output["advantages"].tolist())
        self.assertEqual(masked, [[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]])
        output = dict(completion_mask=torch.ones(6, 2, dtype=torch.long), advantages=torch.ones(6))
        apply_rejection(output, [0], 4, offset=6, total=12)  # group 0 lies wholly on rank 0
        self.assertEqual(output["advantages"].tolist(), [1] * 6)
        with self.assertRaises(ValueError):
            apply_rejection(output, [3], 4, offset=6, total=12)
        with self.assertRaises(ValueError):
            apply_rejection(output, [], 4, offset=8, total=12)

    def test_masks_exactly_the_rejected_group(self):
        import torch
        output = dict(completion_mask=torch.ones(6, 3, dtype=torch.long), advantages=torch.arange(6.) + 1)
        apply_rejection(output, [1], 2)
        self.assertEqual(output["completion_mask"].sum(dim=1).tolist(), [3, 3, 0, 0, 3, 3])
        self.assertEqual(output["advantages"].tolist(), [1, 2, 0, 0, 5, 6])
        with self.assertRaises(ValueError):
            apply_rejection(output, [3], 2)


class SuiteTests(unittest.TestCase):
    def test_frozen_suite_loads_and_splits_by_template(self):
        train, test = load_suite(SUITE)
        self.assertEqual((len(train), len(test)), (180, 60))
        self.assertFalse({r["template"] for r in train} & {r["template"] for r in test})
        for split in (train, test):
            self.assertEqual(sum(r["character"] == "rei" for r in split), len(split) // 2)

    def test_per_character_rows(self):
        train, test = load_suite(SUITE)
        for character in CHARACTERS:
            mine = select_character(train, character), select_character(test, character)
            self.assertEqual((len(mine[0]), len(mine[1])), (90, 30))
            self.assertTrue(all(r["character"] == character for split in mine for r in split))
        self.assertIs(select_character(train, None), train)
        with self.assertRaises(ValueError):
            select_character(train, "shinji")

    def test_v4_training_settings(self):
        train = load_train_script()
        self.assertEqual(train.SCALE_REWARDS, "batch")
        self.assertEqual(train.REWARD_SPEC["name"], "style-margin-v4")
        self.assertEqual(PROTOCOL, "style-v4")

    def test_suite_is_regenerated_byte_for_byte(self):
        import scripts.build_style_suite as build
        train, test, _ = build.build()
        with tempfile.TemporaryDirectory() as tmp:
            dump(Path(tmp) / "train.json", train)
            dump(Path(tmp) / "test.json", test)
            frozen = json.loads((SUITE / "freeze.json").read_text(encoding="utf-8"))
            self.assertEqual(digest(Path(tmp) / "train.json"), frozen["files"]["train.json"])
            self.assertEqual(digest(Path(tmp) / "test.json"), frozen["files"]["test.json"])
            self.assertEqual(frozen["persona_sha256"], persona_identity())

    def test_tampered_suite_and_unbound_replies_are_refused(self):
        train, test = load_suite(SUITE)
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite"
            suite.mkdir()
            for name in ("train.json", "test.json", "freeze.json"):
                (suite / name).write_bytes((SUITE / name).read_bytes())
            replies = Path(tmp) / "base"
            replies.mkdir()
            dump(replies / "replies.json", {r["id"]: "……嗯。" for r in train + test})
            dump(replies / "freeze.json", dict(replies_sha256=digest(replies / "replies.json"),
                                               suite_freeze_sha256=digest(suite / "freeze.json")))
            self.assertEqual(len(load_base_replies(replies, suite)), 240)
            (suite / "train.json").write_bytes((SUITE / "train.json").read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_suite(suite)
            (suite / "train.json").write_bytes((SUITE / "train.json").read_bytes())
            other = Path(tmp) / "other"
            other.mkdir()
            dump(other / "replies.json", {"x": "y"})
            dump(other / "freeze.json", dict(replies_sha256=digest(other / "replies.json"),
                                             suite_freeze_sha256="0" * 64))
            with self.assertRaisesRegex(ValueError, "different suite"):
                load_base_replies(other, suite)


if __name__ == "__main__":
    unittest.main()
