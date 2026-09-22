"""Adversarial synthetic tests for source visibility, split leakage and judge parsing."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from characore import contexts, judge


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = "访客：可以进门吗？\n守卫：请先出示许可。\n"
        (self.root / "source.txt").write_bytes(raw.encode("utf-8"))
        self.row = dict(id="cut", family="gate", split_group="gate_event", character="守卫",
                        source_path="source.txt", cache_path="source.txt",
                        source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        source_revision="pinned", source_kind="upstream_edited_dialogue",
                        status="dev_review", visible_lines=[1], target_line=2,
                        upstream_split="unspecified", exposure="development_review",
                        known="SECRET_KNOWN", unknown="SECRET_UNKNOWN", action_required=True)
        contexts.write_json(self.root / "plan.json", {"version": "fixture"})
        self.write_index()

    def write_index(self):
        (self.root / "index.jsonl").write_text(json.dumps(self.row, ensure_ascii=False) + "\n", encoding="utf-8")

    def build(self, name="snapshot"):
        output = self.root / name
        contexts.build_development(self.root, self.root / "index.jsonl", self.root / "plan.json", output)
        return output

    def test_materialization_separates_hidden_answer_and_metadata(self):
        output = self.build()
        row = contexts.load_contexts(output)["dev"][0]
        prompt = json.dumps(contexts.policy_messages(row), ensure_ascii=False)
        self.assertIn("可以进门吗", prompt)
        for secret in ["请先出示许可", "SECRET", "source.txt", "gate_event"]:
            self.assertNotIn(secret, prompt)
        refs = contexts.read_json(output / "references.json")
        self.assertIn("请先出示许可", str(refs))
        with self.assertRaisesRegex(ValueError, "not a formal"):
            contexts.load_contexts(output, require_formal=True)

    def test_hash_and_future_line_rejected_before_output(self):
        self.row["visible_lines"] = [1, 2]
        self.write_index()
        with self.assertRaisesRegex(ValueError, "future"):
            self.build()
        self.assertFalse((self.root / "snapshot").exists())
        self.row["visible_lines"] = [1]
        self.write_index()
        (self.root / "source.txt").write_text("tamper", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()

    def test_manifest_cannot_drop_dev_hash(self):
        output = self.build()
        manifest = contexts.read_json(output / "snapshot.json")
        del manifest["files"]["dev.json"]
        contexts.write_json(output / "snapshot.json", manifest)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            contexts.load_contexts(output)

    def test_dev_tamper_rejected(self):
        output = self.build()
        with (output / "dev.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Snapshot changed"):
            contexts.load_contexts(output)

    def test_group_source_and_exposure_leakage(self):
        output = self.build()
        row = contexts.load_contexts(output)["dev"][0]
        sources = contexts.read_json(output / "sources.json")
        train = copy.deepcopy(row)
        train.update(id="other", split="train", family="different")
        train["visible_turns"][0]["text"] = "other input"
        with self.assertRaisesRegex(ValueError, "leakage"):
            contexts.validate_splits({"train": [train], "dev": [row], "test": []}, sources)
        test = dict(row, split="test")
        with self.assertRaisesRegex(ValueError, "exposure"):
            contexts.validate_splits({"train": [], "dev": [], "test": [test]}, sources)
        train.update(upstream_split="test")
        with self.assertRaisesRegex(ValueError, "holdout"):
            contexts.validate_splits({"train": [train], "dev": [], "test": []}, sources)

    def test_narration_and_private_thought_require_review(self):
        raw = "旁白：天晴。守卫已经放行。\n守卫：请先出示许可。\n"
        (self.root / "source.txt").write_bytes(raw.encode("utf-8"))
        self.row["source_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        self.write_index()
        with self.assertRaisesRegex(ValueError, "explicit reviewed span"):
            self.build()
        self.row["visible_spans"] = {"1": "天晴。"}
        self.write_index()
        row = contexts.load_contexts(self.build("safe"))["dev"][0]
        self.assertNotIn("已经放行", str(contexts.policy_messages(row)))


class JudgeTests(unittest.TestCase):
    def good(self):
        dimension = dict(status="scored", score=3, reason="引用当前公开输入", evidence_ids=["cut:L1"])
        return dict(protocol=judge.PROTOCOL, winner="tie", reason="同样可行",
                    preference_evidence_ids=["cut:L1"],
                    scores={side: {dim: copy.deepcopy(dimension) for dim in judge.DIMS}
                            for side in ("A", "B")})

    def test_failures_are_not_insufficient_or_zero_rewards(self):
        parsed = judge.parse_response("not json", ["cut:L1"])
        self.assertEqual(parsed["call_status"], "parse_error")
        self.assertIsNone(parsed["judgement"])
        response = self.good()
        response["winner"] = "insufficient"
        response["scores"]["A"]["C"].update(status="insufficient", score=None)
        self.assertEqual(judge.parse_response(json.dumps(response), ["cut:L1"])["call_status"], "ok")
        self.assertIsNone(judge.composite(response["scores"]["A"]))

    def test_unknown_citations_booleans_missing_dims_and_na_rejected(self):
        for mutate in (
            lambda r: r["scores"]["A"]["C"].update(score=True),
            lambda r: r["scores"]["A"]["S"].update(evidence_ids=["invented"]),
            lambda r: r["scores"]["B"].pop("Q"),
            lambda r: r["scores"]["A"]["C"].update(status="not_applicable", score=None),
            lambda r: r["scores"]["A"]["A"].update(status="not_applicable", score=None),
        ):
            response = self.good()
            mutate(response)
            self.assertEqual(judge.parse_response(json.dumps(response), ["cut:L1"], True)["call_status"], "parse_error")

    def test_swapping_and_hidden_metadata_not_sent(self):
        row = dict(id="cut", character="守卫", action_required=False,
                   visible_turns=[dict(source_line=1, speaker="访客", text="请进门")],
                   reference="SECRET", method="SECRET")
        candidates = {"A": "回答甲", "B": "回答乙"}
        ab = judge.make_request(row, candidates, {}, False)
        ba = judge.make_request(row, candidates, {}, True)
        self.assertEqual(json.loads(ab["messages"][1]["content"])["candidate_A"], "回答甲")
        self.assertEqual(json.loads(ba["messages"][1]["content"])["candidate_A"], "回答乙")
        self.assertNotIn("SECRET", json.dumps(ab["messages"]))
        self.assertEqual(ba["display_to_original"]["A"], "B")


if __name__ == "__main__":
    unittest.main()
