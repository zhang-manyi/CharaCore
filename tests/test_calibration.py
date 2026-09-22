"""Regression tests for blinding, version binding and offline human materials."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from characore import calibration, contexts, judge


class JudgeReviewTests(unittest.TestCase):
    def response(self):
        dimension = dict(status="scored", score=3, reason="Visible support", evidence_ids=["E1"])
        return dict(protocol=judge.PROTOCOL, winner="A", reason="Supported difference",
                    preference_evidence_ids=["E1", "candidate:B"],
                    scores={side: {dim: copy.deepcopy(dimension) for dim in judge.DIMS}
                            for side in ("A", "B")})

    def test_neutral_evidence_ids_do_not_reveal_cutoff(self):
        row = dict(id="SECRET_future_topic", character="Guard", action_required=False,
                   visible_turns=[dict(source_line=7, speaker="Visitor", text="May I enter?")])
        request = judge.make_request(row, {"A": "Show a pass.", "B": "You entered yesterday."}, {})
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["evidence"][0]["id"], "E1")
        self.assertNotIn("SECRET", json.dumps(request["messages"]))
        self.assertEqual(request["allowed_evidence_ids"], ["E1", "candidate:A", "candidate:B"])
        old = judge.make_request(row, {"A": "First", "B": "Second"}, {}, protocol=judge.LEGACY_PROTOCOL)
        self.assertEqual(old["messages"][0]["content"], judge.SYSTEM_V02)

    def test_duplicate_json_keys_and_invalid_shapes_fail_closed(self):
        raw = json.dumps(self.response())
        duplicate = raw.replace('"winner": "A"', '"winner": "B", "winner": "A"')
        for value in (duplicate, "null", "[]", "42", '{"scores": []}'):
            result = judge.parse_response(value, ["E1", "candidate:B"])
            self.assertEqual(result["call_status"], "parse_error")
            self.assertIsNone(result["judgement"])

    def test_pair_citations_and_protocol_are_bound(self):
        response = self.response()
        self.assertEqual(judge.parse_response(json.dumps(response), ["E1", "candidate:B"])["call_status"], "ok")
        for refs in ([], ["secret_reference"], "E1"):
            changed = copy.deepcopy(response)
            changed["preference_evidence_ids"] = refs
            self.assertEqual(judge.parse_response(json.dumps(changed), ["E1"])["call_status"], "parse_error")
        legacy = copy.deepcopy(response)
        legacy.pop("preference_evidence_ids")
        legacy["protocol"] = judge.LEGACY_PROTOCOL
        self.assertEqual(judge.parse_response(json.dumps(legacy), ["E1"])["call_status"], "parse_error")
        self.assertEqual(judge.parse_response(json.dumps(legacy), ["E1"], protocol=judge.LEGACY_PROTOCOL)["call_status"], "ok")

    def test_required_action_can_be_insufficient_but_never_na(self):
        response = self.response()
        response["scores"]["A"]["A"].update(status="insufficient", score=None, evidence_ids=[])
        result = judge.parse_response(json.dumps(response), ["E1", "candidate:B"], True)
        self.assertEqual(result["call_status"], "ok")
        self.assertIsNone(judge.composite(result["judgement"]["scores"]["A"]))
        response["scores"]["A"]["A"]["status"] = "not_applicable"
        self.assertEqual(judge.parse_response(json.dumps(response), ["E1", "candidate:B"], True)["call_status"], "parse_error")


class HumanPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = "访客：可以进门吗？\n守卫：隐藏参考答案。\n"
        (self.root / "source.txt").write_bytes(raw.encode("utf-8"))
        row = dict(id="hidden_topic", family="family", split_group="group", character="守卫",
                   source_path="source.txt", cache_path="source.txt",
                   source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                   source_revision="pinned", source_kind="upstream", status="dev_review",
                   visible_lines=[1], target_line=2, upstream_split="unspecified",
                   exposure="development_review", known="SECRET", unknown="SECRET",
                   action_required=True)
        (self.root / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        contexts.write_json(self.root / "plan.json",
                            dict(judge_protocol=judge.PROTOCOL, rubric={},
                                 calibration=dict(planned_unique_pairs=2)))
        self.suite = self.root / "suite"
        contexts.build_development(self.root, self.root / "index.jsonl", self.root / "plan.json", self.suite)
        self.cases = [
            dict(id="secret_expected_a", context_id="hidden_topic", expected="A",
                 reason="SECRET", focus="SECRET", candidates={"A": "请出示许可。", "B": "你昨天已进来。"}),
            dict(id="secret_expected_tie", context_id="hidden_topic", expected="tie",
                 reason="SECRET", focus="SECRET", candidates={"A": "请稍等。", "B": "稍等一下。"}),
        ]
        self.cases_path = self.root / "cases.json"
        contexts.write_json(self.cases_path, self.cases)

    def test_packet_is_blind_blank_bound_and_swapped(self):
        output = self.root / "packet"
        manifest = calibration.build_packet(self.suite, self.cases_path, output)
        self.assertEqual(manifest["human_reviews_completed"], 0)
        self.assertEqual(manifest["calls_executed"], 0)
        self.assertFalse(manifest["calibration_passed"])
        blind = (output / "human" / "review.json").read_text(encoding="utf-8")
        for secret in ("hidden_topic", "secret_expected", "SECRET", "隐藏参考答案", "source.txt"):
            self.assertNotIn(secret, blind)
        answers = contexts.read_json(output / "human" / "reviewer_1.json")
        self.assertTrue(all(item["winner"] is None for item in answers["annotations"]))
        requests = contexts.read_json(output / "judge" / "requests.json")
        for ab, ba in zip(requests[::2], requests[1::2]):
            a = json.loads(ab["messages"][1]["content"])
            b = json.loads(ba["messages"][1]["content"])
            self.assertEqual(a["candidate_A"], b["candidate_B"])
            self.assertEqual(a["candidate_B"], b["candidate_A"])
        calibration.verify_packet(output)
        with self.assertRaisesRegex(ValueError, "exists"):
            calibration.build_packet(self.suite, self.cases_path, output)
        with (output / "human" / "review.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            calibration.verify_packet(output)

    def test_duplicate_pairs_or_non_dev_context_rejected_before_output(self):
        for mutation in ("duplicate", "non_dev"):
            cases = copy.deepcopy(self.cases)
            if mutation == "duplicate":
                cases[1]["candidates"] = {"A": cases[0]["candidates"]["B"], "B": cases[0]["candidates"]["A"]}
            else:
                cases[1]["context_id"] = "test_only"
            contexts.write_json(self.cases_path, cases)
            output = self.root / mutation
            with self.assertRaises(ValueError):
                calibration.build_packet(self.suite, self.cases_path, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
