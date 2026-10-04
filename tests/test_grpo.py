"""Style reward, judge protocol, suite freeze and group-rejection checks. Standard library only
(apply_rejection uses torch when it is importable)."""
import json
from pathlib import Path
import tempfile
import unittest

from characore.judge import (ATTR_PROTOCOL, PROTOCOL, make_attribution_request, make_request,
                             parse_attribution, parse_response)
from characore.persona import CHARACTERS, persona_identity, policy_messages
from characore.protocol import digest, dump
from characore.stub_judge import StubJudge
from characore.style_data import load_base_replies, load_suite
from characore.style_rewards import (StyleReward, clean_reply, hard_violation, pair_outcome,
                                     rule_metrics, style_score)
from characore.style_trainer import apply_rejection

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "experiments/style_v1"
ROW = dict(id="T01-L1-rei", template="T01", character="rei", situation="放学后的教室。", speaker="同班同学",
           line="你带伞了吗？")
IDS = ["E1", "E2", "E3", "candidate:A", "candidate:B"]


def judgement(winner="A", r=3, protocol=PROTOCOL):
    dim = lambda score: dict(status="scored", score=score, reason="ok", evidence_ids=["E1"])
    side = lambda: {"P": dim(3), "R": dim(r), "N": dim(3)}
    return dict(protocol=protocol, winner=winner, scores={"A": side(), "B": side()}, reason="ok",
                preference_evidence_ids=[] if winner in ("tie", "insufficient") else ["E1"])


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

    def test_clean_reply_strips_only_empty_think(self):
        self.assertEqual(clean_reply("<think>\n\n</think>\n\n……没有。 "), "……没有。")
        self.assertEqual(clean_reply("<think>想</think>没有"), "<think>想</think>没有")

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

    def test_prompt_asks_for_scores_last(self):
        system = make_request(ROW, {"A": "……嗯。", "B": "……不。"})["messages"][0]["content"]
        self.assertIn("protocol, winner, reason, preference_evidence_ids, scores", system)
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


class PairOutcomeTests(unittest.TestCase):
    AB, BA = {"A": "A", "B": "B"}, {"A": "B", "B": "A"}

    def test_mirrored_winners_agree(self):
        self.assertEqual(pair_outcome(call("A", self.AB), call("B", self.BA)), (1.0, False, False))
        self.assertEqual(pair_outcome(call("B", self.AB), call("A", self.BA)), (0.0, False, False))
        self.assertEqual(pair_outcome(call("tie", self.AB), call("tie", self.BA)), (0.5, False, False))

    def test_order_disagreement_is_a_flagged_tie(self):
        # Position-following (A, A) and a tie in one order only both mean the judge cannot separate the pair.
        for ab, ba in ((call("A", self.AB), call("A", self.BA)), (call("tie", self.AB), call("B", self.BA))):
            self.assertEqual(pair_outcome(ab, ba), (0.5, False, True))

    def test_failure_and_insufficient_are_rejected(self):
        for ab, ba, reason in ((call("A", self.AB, status="parse_error"), call("B", self.BA), "AB parse_error"),
                               (call("insufficient", self.AB), call("insufficient", self.BA), "insufficient")):
            with self.assertRaisesRegex(ValueError, reason):
                pair_outcome(ab, ba)

    def test_off_topic_needs_both_orders(self):
        self.assertTrue(pair_outcome(call("B", self.AB, r=1), call("A", self.BA, r=0))[1])
        self.assertFalse(pair_outcome(call("B", self.AB, r=1), call("A", self.BA, r=3))[1])


class StyleRewardTests(unittest.TestCase):
    def reward(self, judge, group=2):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return StyleReward(judge, Path(tmp.name) / "r", group)

    def test_usable_batch_values_and_reuse(self):
        judge = ScriptedJudge("A", "B")  # policy wins in both orders
        reward = self.reward(judge)
        values = reward(**batch(["……没有带。", "……没有带。", "作为AI我不知道", "……base"]))
        win = round(0.7 + 0.3 * style_score("……没有带。", "rei"), 6)
        self.assertEqual(values, [win, win, -1.0, round(0.35 + 0.3 * style_score("……base", "rei"), 6)])
        self.assertEqual(reward.take_rejected(), [])
        # Two identical policy replies share one AB and one BA call.
        self.assertEqual(judge.calls, 2)
        self.assertEqual(reward.last_metrics["reused_requests"], 2)
        self.assertEqual(reward.totals["skipped_identical"], 1)
        with self.assertRaises(RuntimeError):
            reward.take_rejected()

    def test_order_disagreement_scores_tie_and_is_counted(self):
        judge = ScriptedJudge("A", "A")  # winner follows display position: inconsistent
        reward = self.reward(judge)
        values = reward(**batch(["……没有带。", "作为AI我不知道"]))
        self.assertEqual(values, [round(0.35 + 0.3 * style_score("……没有带。", "rei"), 6), -1.0])
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
        self.assertEqual(values[0], round(0.7 + 0.3 * style_score("……没有带。", "rei"), 6))
        self.assertEqual(judge.calls, 3)  # AB, BA, and one retry of the first call

    def test_batch_must_be_whole_consistent_groups(self):
        reward = self.reward(ScriptedJudge())
        with self.assertRaisesRegex(ValueError, "whole groups"):
            reward(**batch(["……没有带。"] * 3))
        other = dict(ROW, id="T01-L2-rei")
        with self.assertRaisesRegex(ValueError, "share one prompt"):
            reward(**batch(["……没有带。", "……没有。"], rows=[ROW, other]))

    def test_stub_judge_is_order_consistent(self):
        reward = self.reward(StubJudge())
        values = reward(**batch(["……没有带。", "哼，笨蛋，我当然带了！"], base="好的，我带了伞，可以借给你。"))
        self.assertEqual(reward.take_rejected(), [])
        self.assertGreater(values[0], values[1])


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch not installed")
class RejectionMaskTests(unittest.TestCase):
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
