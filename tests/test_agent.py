"""Authored task mechanics only; no model, reward or character-quality claims."""
import json
from pathlib import Path
import tempfile
import unittest

from characore.agent import DeliveryTask, ScriptedPolicy, command, messages, parse_output, replay, run_episode


class AgentTests(unittest.TestCase):
    def test_scripted_paths_and_verified_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, rejects, steps in (("success", 0, 6), ("illegal", 2, 8), ("commitment", 1, 7)):
                with self.subTest(name=name):
                    path = Path(tmp) / name
                    trace = run_episode(ScriptedPolicy(name), path)
                    self.assertTrue(trace["metrics"]["task_completed"])
                    self.assertEqual(trace["metrics"]["tool_rejections"], rejects)
                    self.assertEqual(trace["metrics"]["steps"], steps)
                    self.assertEqual(trace["metrics"]["rejected_action_state_mutations"], 0)
                    self.assertEqual(replay(path / "trajectory.json"), trace)
                    self.assertTrue(trace["metrics"]["commitment_kept"])
                    if name == "commitment":
                        row = trace["records"][4]
                        visible = json.loads(row["visible_input"][1]["content"])
                        self.assertEqual(visible["state"]["commitments"][0]["made_turn"], 2)
                        self.assertEqual(visible["events"][1]["result"]["commitment"], "keep_sealed")
                        self.assertEqual(row["tool_result"]["error"], "commitment_violation")

    def test_hidden_clue_disclosure_and_input_copies(self):
        env = DeliveryTask()
        initial = messages(env.observation())
        self.assertNotIn("积水", json.dumps(initial, ensure_ascii=False))
        self.assertFalse(env.step(command("inspect_clue", clue_id="route"))["tool_result"]["ok"])
        env.step(command("inspect_clue", clue_id="dispatch"))
        env.step(command("submit_action", action="promise", promise="keep_sealed"))
        self.assertNotIn("积水", json.dumps(messages(env.observation()), ensure_ascii=False))
        env.step(command("inspect_clue", clue_id="route"))
        self.assertIn("积水", json.dumps(messages(env.observation()), ensure_ascii=False))
        obs = env.observation()
        obs["state"]["known_facts"].clear()
        obs["events"].clear()
        self.assertIn("route", env.state["known_facts"])
        self.assertEqual(len(env.memory), 4)

    def test_preconditions_and_bad_arguments_do_not_mutate(self):
        cases = [command("submit_action", action="promise", promise="keep_sealed"),
                 command("submit_action", action="move", route="north"),
                 command("submit_action", action="deliver"),
                 command("submit_action", action="open"),
                 command("submit_action", action=[]),
                 command("inspect_clue", clue_id=[]),
                 command("query_status", injected="x"), command("unknown")]
        for raw in cases:
            with self.subTest(raw=raw):
                row = DeliveryTask().step(raw)
                self.assertFalse(row["tool_result"]["ok"])
                self.assertEqual(set(row["state_changes"]), {"turn"})

    def test_blocked_route_and_repeat_movement(self):
        env = DeliveryTask()
        for raw in (command("inspect_clue", clue_id="dispatch"),
                    command("submit_action", action="promise", promise="keep_sealed"),
                    command("inspect_clue", clue_id="route")):
            env.step(raw)
        row = env.step(command("submit_action", action="move", route="south"))
        self.assertEqual(row["tool_result"]["error"], "blocked_route")
        self.assertEqual(env.state["location"], "depot")
        env.step(command("submit_action", action="move", route="north"))
        row = env.step(command("submit_action", action="move", route="north"))
        self.assertEqual(row["tool_result"]["error"], "precondition")

    def test_terminal_budget_parse_failures_and_abort(self):
        env = DeliveryTask(2)
        env.step("我已完成递送。")
        self.assertFalse(env.state["delivered"])
        row = env.step("{}")
        self.assertEqual(row["failure"]["kind"], "parse_error")
        self.assertEqual(env.state["status"], "step_limit")
        with self.assertRaises(ValueError):
            env.step(command("query_status"))
        self.assertEqual(env.execute("query_status", {})["error"], "terminal")
        env = DeliveryTask()
        env.step(command("submit_action", action="abort"))
        self.assertEqual(env.state["status"], "aborted")
        for limit in (0, -1, 51, True):
            with self.assertRaises(ValueError):
                DeliveryTask(limit)

    def test_strict_json(self):
        for raw in ('{"speech":"a","speech":"b","tool":"query_status","arguments":{}}',
                    '{"speech":"a","tool":"query_status","arguments":[],"other":1}',
                    '{"speech":"a","tool":"query_status","arguments":{"x":NaN}}'):
            with self.assertRaises(ValueError):
                parse_output(raw)

    def test_policy_failure_record_and_no_overwrite(self):
        class Broken:
            metadata = {"kind": "synthetic_failure_test"}

            def __call__(self, visible):
                raise RuntimeError("local inference failed")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed"
            trace = run_episode(Broken(), path)
            self.assertEqual(trace["metrics"]["policy_failures"], 1)
            self.assertEqual(trace["final_state"]["status"], "policy_error")
            replay(path / "trajectory.json")
            with self.assertRaises(FileExistsError):
                run_episode(Broken(), path)

    def test_replay_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original"
            trace = run_episode(ScriptedPolicy(), path)
            trace["records"][0]["tool_result"]["ok"] = False
            changed = Path(tmp) / "tampered.json"
            changed.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                replay(changed)


if __name__ == "__main__":
    unittest.main()
