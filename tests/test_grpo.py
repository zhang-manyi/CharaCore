"""Synthetic fixtures only. Rewards, group validation and the frozen-suite gate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from characore.agent import TASK_ID, command, dump
from characore.grpo_data import load_training_suite
from characore.grpo_rewards import (ActionReward, REWARD_SPEC, UnusableReward, pair_reward,
                                    restore, transition, validate_groups)
from characore.judge import DIMS, PROTOCOL
from characore.judge_runner import identity
from characore.protocol import digest, read_json
from characore.stub_judge import StubJudge


def judged(winner="A"):
    dim = dict(status="scored", score=3, reason="Synthetic fixture only", evidence_ids=["E1"])
    obj = dict(protocol=PROTOCOL, winner=winner, reason="Synthetic fixture only",
               preference_evidence_ids=["E1"],
               scores={side: {d: copy.deepcopy(dim) for d in DIMS} for side in ("A", "B")})
    return dict(call_status="ok", judgement=obj, raw=json.dumps(obj))


class RewardTests(unittest.TestCase):
    def test_order_mapping_tie_and_failures(self):
        self.assertEqual(pair_reward(judged("A"), judged("B")), 1)
        self.assertEqual(pair_reward(judged("B"), judged("A")), 0)
        self.assertEqual(pair_reward(judged("tie"), judged("tie")), .5)
        missing_dim = judged()
        missing_dim["judgement"]["scores"]["A"]["C"].update(status="insufficient", score=None)
        for a, b in [(judged("A"), judged("A")), (judged("insufficient"), judged()),
                     (dict(call_status="parse_error", judgement=None), judged()), (missing_dim, judged())]:
            with self.assertRaises(UnusableReward):
                pair_reward(a, b)

    def test_no_silent_zero_and_equal_groups_reported_not_rejected(self):
        for values in ([None, 1], [float("nan"), 1], [float("inf"), 1], [True, 1], [1]):
            with self.assertRaises(UnusableReward):
                validate_groups(values, 2)
        self.assertEqual(validate_groups([0, 1], 2), ([0, 1], 0))
        # An all-equal group is a no-op update, not corrupt data: counted, not raised.
        self.assertEqual(validate_groups([.5, .5], 2), ([.5, .5], 1))

    def test_actual_transition_and_no_future_input(self):
        visible, row, progress = transition([], command("inspect_clue", clue_id="dispatch"))
        self.assertEqual(progress, 1)
        self.assertNotIn("积水", json.dumps(visible, ensure_ascii=False))
        prefix = [command("inspect_clue", clue_id="dispatch"),
                  command("submit_action", action="promise", promise="keep_sealed")]
        _, row, progress = transition(prefix, prefix[-1])
        self.assertTrue(row["tool_result"]["ok"])
        self.assertEqual(progress, 0)
        _, row, progress = transition(prefix, command("submit_action", action="open"))
        self.assertEqual(row["tool_result"]["error"], "commitment_violation")
        with self.assertRaises(ValueError):
            restore([command("submit_action", action="deliver")])

    def test_restore_matches_evaluation_horizon(self):
        """A training checkpoint must show the same steps_remaining as evaluation."""
        env = restore([])
        self.assertEqual(env.observation()["steps_remaining"], env.max_steps)

    def test_reward_adapter_executes_calls_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = ActionReward(None, Path(tmp) / 'reward', {}, 2)
            with patch('characore.grpo_rewards.call_judge',
                       side_effect=[{'final': judged('A')}, {'final': judged('B')}]):
                values = callback([command('submit_action', action='deliver'),
                                   command('inspect_clue', clue_id='dispatch')],
                                  [[], []], [command('query_status')] * 2)
            self.assertEqual(values, [-1, 1])
            self.assertTrue((Path(tmp) / 'reward/batch_0001/rewards.json').exists())
            with patch('characore.grpo_rewards.call_judge',
                       return_value={'final': dict(call_status='inference_error')}):
                # Completions must differ from the anchor, or the tie path skips the judge.
                with self.assertRaises(UnusableReward):
                    callback([command('inspect_clue', clue_id='dispatch')] * 2, [[], []],
                             [command('query_status')] * 2)
            self.assertTrue((Path(tmp) / 'reward/batch_0002/failure.json').exists())
            with self.assertRaisesRegex(UnusableReward, 'share one'):
                callback([command('query_status')] * 2,
                         [[], [command('inspect_clue', clue_id='dispatch')]],
                         [command('query_status')] * 2)

    def test_identical_candidate_scores_tie_without_a_judge_call(self):
        """Reproducing the anchor must not abort the batch or spend a judge call."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = ActionReward(None, Path(tmp) / 'reward', {}, 2)
            anchor = command('query_status')
            with patch('characore.grpo_rewards.call_judge',
                       side_effect=AssertionError("must not call the judge")):
                values = callback([anchor] * 2, [[], []], [anchor] * 2)
            expected = REWARD_SPEC["judge_weight"] * REWARD_SPEC["pairwise_values"]["tie"]
            self.assertEqual(values, [expected] * 2)


class StubJudgeTests(unittest.TestCase):
    def test_emits_protocol_valid_order_independent_judgements(self):
        with tempfile.TemporaryDirectory() as tmp:
            reward = ActionReward(StubJudge(), Path(tmp) / 'reward', read_json(
                Path(__file__).resolve().parents[1] / 'experiments/agent_v1/evaluation_plan.json')['rubric'], 2)
            # A legal action against an illegal one must be ranked, not rejected.
            values = reward([command('inspect_clue', clue_id='dispatch'),
                             command('submit_action', action='deliver')],
                            [[], []], [command('query_status')] * 2)
            self.assertEqual(len(values), 2)
            self.assertGreater(values[0], values[1])


class TrainingGateTests(unittest.TestCase):
    def make_suite(self, root, scope='development_only'):
        for split, prefix in [('train', []), ('eval', [command('inspect_clue', clue_id='dispatch')])]:
            dump(root / f'{split}.json', [dict(id=split, split=split, exposure='development',
                                               family=TASK_ID, source=TASK_ID, prefix=prefix,
                                               anchor=command('query_status'))])
        dump(root / 'freeze.json', dict(task_id=TASK_ID, evaluation_scope=scope,
             reward_spec_sha256=identity(REWARD_SPEC),
             files={n: digest(root / n) for n in ('train.json', 'eval.json')}))

    def test_development_is_not_holdout_and_freeze_is_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_suite(root)
            self.assertEqual(len(load_training_suite(root)[0]), 1)
            path = root / 'freeze.json'
            obj = read_json(path)
            obj['evaluation_scope'] = 'independent_holdout'
            path.write_text(json.dumps(obj), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'relabel'):
                load_training_suite(root)
            (root / 'train.json').write_text('[]', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_training_suite(root)

    def test_reward_spec_change_invalidates_the_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_suite(root)
            with patch.dict(REWARD_SPEC, {'judge_weight': .5}):
                with self.assertRaisesRegex(ValueError, 'reward'):
                    load_training_suite(root)

    def test_committed_suite_passes_preflight(self):
        suite = Path(__file__).resolve().parents[1] / 'experiments/agent_v1/suite'
        train, evaluation = load_training_suite(suite)
        self.assertTrue(train and evaluation)
        depths = {len(row['prefix']) for row in evaluation}
        self.assertGreater(len(depths), 1, "eval split must cover more than one depth")


if __name__ == '__main__':
    unittest.main()
