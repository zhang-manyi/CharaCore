"""Small synthetic fixtures test contracts, not role-playing quality."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from characore import protocol
from scripts import train_dpo


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.suite = Path(self.folder.name)
        self.profiles = {"keeper": {"name": "守门人", "role": "守卫",
            "stable_preferences": "按通行许可行动。", "style": "简洁。"}}
        base = dict(id="train-allow", family="train-gate", character="keeper",
                    goal="检查通行许可", relationship="访客", history="未曾见面",
                    emotion="平静", situation="访客持有有效许可", user="请开门",
                    candidates={"a": "核验许可后开门。", "b": "无需核验就开门。"},
                    preference="a", evidence="应先核验许可")
        self.train = [base,
            dict(base, id="train-tie", family="train-tie", situation="两种核验方式均有效",
                 preference="tie"),
            dict(base, id="train-unknown", family="train-unknown", situation="许可状态未知",
                 preference="insufficient")]
        self.evaluation = [dict(base, id="test-expired", family="test-gate",
                                situation="许可已经过期")]
        self.write("profiles.json", self.profiles)
        self.write("train.json", self.train)
        self.write("eval.json", self.evaluation)
        self.write("evaluation_plan.json", {"version": "unit-test-only"})
        self.write("freeze.json", {"files": {name: protocol.digest(self.suite / name)
            for name in ["profiles.json", "eval.json", "evaluation_plan.json"]}})
        self.write("train_manifest.json", {
            "evaluation_freeze_sha256": protocol.digest(self.suite / "freeze.json"),
            "train_sha256": protocol.digest(self.suite / "train.json")})

    def write(self, name, data):
        (self.suite / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_loads_explicit_suite_and_rejects_family_leakage(self):
        _, train, evaluation = protocol.load_suite(self.suite)
        with self.assertRaisesRegex(ValueError, "leakage"):
            protocol.validate(train, [dict(evaluation[0], family=train[0]["family"])])

    def test_tampered_evaluation_is_rejected(self):
        with (self.suite / "eval.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Frozen evaluation changed"):
            protocol.load_suite(self.suite)

    def test_tampered_training_is_rejected(self):
        with (self.suite / "train.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Training manifest mismatch"):
            protocol.load_suite(self.suite)

    def test_labels_and_candidates_never_enter_prompts(self):
        row = copy.deepcopy(self.train[0])
        row["evidence"] = "PRIVATE_EVIDENCE_SENTINEL"
        row["candidates"] = {"a": "PRIVATE_A_SENTINEL", "b": "PRIVATE_B_SENTINEL"}
        prompt = json.dumps(protocol.prompt_messages(row, self.profiles), ensure_ascii=False)
        self.assertNotIn("PRIVATE_", prompt)
        self.assertIn(row["situation"], prompt)
        self.assertEqual(protocol.prompt_messages(row, self.profiles),
                         protocol.prompt_messages(dict(row, preference="b"), self.profiles))

    def test_non_strict_preferences_excluded_and_order_preserved(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kw: "context")
        records = protocol.dpo_records(self.train, self.profiles, tokenizer)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["chosen"], self.train[0]["candidates"]["a"])
        reversed_row = dict(self.train[0], preference="b")
        reversed_records = protocol.dpo_records([reversed_row], self.profiles, tokenizer)
        self.assertEqual(reversed_records[0]["chosen"], self.train[0]["candidates"]["b"])

    def test_identical_candidates_cannot_be_strict(self):
        bad = copy.deepcopy(self.train)
        bad[0]["candidates"]["b"] = bad[0]["candidates"]["a"]
        with self.assertRaisesRegex(ValueError, "Identical"):
            protocol.validate(bad, self.evaluation)

    def test_duplicate_scenario_is_rejected(self):
        duplicate = dict(self.train[0], id="other", family="other")
        with self.assertRaisesRegex(ValueError, "Duplicate scenario input"):
            protocol.validate(self.train, [duplicate])

    def test_cli_requires_explicit_suite_before_creating_output(self):
        output = self.suite / "must-not-exist"
        with patch("sys.argv", ["train_dpo", "--base", "local-model", "--output", str(output)]):
            with self.assertRaises(SystemExit) as error:
                train_dpo.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
