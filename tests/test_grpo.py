"""Synthetic fixtures only. Never write completed labels into real calibration materials."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from characore.agent import TASK_ID, command, dump
from characore.calibration import build_packet
from characore.calibration_report import read, summarize
from characore.grpo_data import load_training_suite, verify_calibration
from characore.grpo_rewards import (ActionReward, REWARD_SPEC, UnusableReward, pair_reward,
                                    restore, transition, validate_groups)
from characore.judge import DIMS, PROTOCOL
from characore.judge_runner import call_judge, identity
from characore.protocol import digest
import test_calibration


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
        for a,b in [(judged("A"), judged("A")), (judged("insufficient"), judged()),
                    (dict(call_status="parse_error", judgement=None), judged()), (missing_dim, judged())]:
            with self.assertRaises(UnusableReward):
                pair_reward(a,b)

    def test_no_silent_zero_or_tie_breaking(self):
        for values in ([None,1], [float("nan"),1], [float("inf"),1], [True,1], [.5,.5], [1]):
            with self.assertRaises(UnusableReward):
                validate_groups(values,2)
        self.assertEqual(validate_groups([0,1],2),[0,1])

    def test_actual_transition_and_no_future_input(self):
        visible,row,progress = transition([], command("inspect_clue", clue_id="dispatch"))
        self.assertEqual(progress,1)
        self.assertNotIn("积水", json.dumps(visible,ensure_ascii=False))
        prefix=[command("inspect_clue",clue_id="dispatch"),
                command("submit_action",action="promise",promise="keep_sealed")]
        _, row, progress = transition(prefix,prefix[-1])
        self.assertTrue(row["tool_result"]["ok"])
        self.assertEqual(progress,0)
        _, row, progress = transition(prefix,command("submit_action",action="open"))
        self.assertEqual(row["tool_result"]["error"],"commitment_violation")
        with self.assertRaises(ValueError):
            restore([command("submit_action",action="deliver")])

    def test_reward_adapter_executes_calls_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback=ActionReward(None,Path(tmp)/'reward',{},2)
            with patch('characore.grpo_rewards.call_judge',side_effect=[{'final':judged('A')},{'final':judged('B')}]):
                values=callback([command('submit_action',action='deliver'),command('inspect_clue',clue_id='dispatch')],
                                [[],[]],[command('query_status')]*2)
            self.assertEqual(values,[-1,1])
            self.assertTrue((Path(tmp)/'reward/batch_0001/rewards.json').exists())
            with patch('characore.grpo_rewards.call_judge',return_value={'final':dict(call_status='inference_error')}):
                with self.assertRaises(UnusableReward):
                    callback([command('query_status')]*2,[[],[]],[command('query_status')]*2)
            self.assertTrue((Path(tmp)/'reward/batch_0002/failure.json').exists())
            with self.assertRaisesRegex(UnusableReward,'share one'):
                callback([command('query_status')]*2,[[],[command('inspect_clue',clue_id='dispatch')]],
                         [command('query_status')]*2)


class CalibrationReportTests(unittest.TestCase):
    def setUp(self):
        fixture=test_calibration.HumanPacketTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root=fixture.root
        self.packet=self.root/'packet'
        build_packet(fixture.suite,fixture.cases_path,self.packet)
        self.requests=read(self.packet/'judge/requests.json')
        self.results=self.root/'results'
        self.results.mkdir()

    def model_results(self):
        finals={}
        for request in self.requests:
            result=judged(request['display_to_original']['A'])
            path=self.results/request['id'];path.mkdir()
            dump(path/'result.json',dict(id=request['id'],request_sha256=identity(request),
                                        attempts=[result],final=result,judge_identity='synthetic-judge'))
            finals[request['id']]=result
        return finals

    def synthetic_humans(self,finals):
        folder=self.root/'human_results';folder.mkdir()
        mapping=read(self.packet/'audit/mapping_and_expectations.json')
        locked={}
        annotations=[]
        for i in (1,2):
            form=read(self.packet/f'human/reviewer_{i}.json')
            form.update(reviewer=f'synthetic-reviewer-{i}',completed_at='test-only')
            for ann in form['annotations']:
                winner=mapping[ann['id']]['display_to_original']['A']
                ann.update({k:v for k,v in judged(winner)['judgement'].items() if k!='protocol'})
                ann['review_status']='completed'
            dump(folder/f'reviewer_{i}.json',form)
            locked[f'reviewer_{i}.json']=digest(folder/f'reviewer_{i}.json')
            annotations=form['annotations']
        dump(folder/'adjudication.json',dict(reviewer='synthetic-adjudicator',completed_at='test-only',
            reviewer_files_sha256=locked,annotations=annotations,
            model_citation_audit=dict(judge_identity='synthetic-judge',reviewed_request_ids=sorted(finals),
                                     results_sha256=identity(finals),completed=True,unsupported_citations=0)))
        return folder

    def test_pending_without_human_or_model_never_passes(self):
        report=summarize(self.packet)
        self.assertFalse(report['calibration_passed'])
        self.assertEqual(report['uncalled'],4)
        self.assertIsNone(report['human_agreement']['value'])
        self.model_results()
        self.assertFalse(summarize(self.packet,self.results)['calibration_passed'])

    def test_synthetic_pass_scope_and_human_binding(self):
        finals=self.model_results()
        human=self.synthetic_humans(finals)
        report=summarize(self.packet,self.results,human)
        self.assertTrue(report['calibration_passed'])
        self.assertEqual(report['order_consistency']['value'],1)
        with self.assertRaisesRegex(ValueError,'cannot authorize'):
            verify_calibration(self.packet,self.results,human)
        form=human/'reviewer_1.json'
        form.write_text(form.read_text(encoding='utf8')+' ',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'locked'):
            summarize(self.packet,self.results,human)

    def test_failed_requests_stay_in_denominator_and_raw_binding(self):
        self.model_results()
        path=next(self.results.glob('*/result.json'));result=read(path)
        failure=dict(call_status='parse_error',judgement=None,raw='not JSON')
        result.update(attempts=[failure],final=failure)
        path.write_text(json.dumps(result),encoding='utf8')
        report=summarize(self.packet,self.results)
        self.assertEqual(report['final_valid'],dict(numerator=3,denominator=4,value=.75))
        result['request_sha256']='tampered'
        path.write_text(json.dumps(result),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'binding'):
            summarize(self.packet,self.results)

    def test_rejects_unknown_and_repeated_result_ids(self):
        self.model_results()
        source=next(self.results.glob('*/result.json'))
        other=self.results/'duplicate';other.mkdir()
        (other/'result.json').write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError,'duplicate'):
            summarize(self.packet,self.results)

    def test_runner_retry_and_input_budget(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return 'test'
            def __call__(self,*args,**kwargs): return {'input_ids':[1,2]}
        class Policy:
            tokenizer=Tokenizer()
            metadata={k:'test' for k in ('model_files_sha256','packages','source_sha256','device','dtype',
                                       'quantized_4bit','seed','do_sample','max_new_tokens','enable_thinking')}
            def __init__(self): self.calls=0
            def __call__(self,visible):
                self.calls+=1
                return ('bad' if self.calls==1 else judged()['raw']),{}
        policy=Policy()
        result=call_judge(self.requests[0],policy,self.root/'retry')
        self.assertEqual([r['call_status'] for r in result['attempts']],['parse_error','ok'])
        result=call_judge(self.requests[0],policy,self.root/'budget',input_limit=1)
        self.assertEqual(result['final']['call_status'],'inference_error')
        self.assertEqual(policy.calls,2)


class TrainingGateTests(unittest.TestCase):
    def make_suite(self,root,scope='development_only'):
        approval=dict(reviewer='synthetic-human',completed_at='test',approved=True,
                      reward_spec_sha256=identity(REWARD_SPEC))
        for split,prefix in [('train',[]),('eval',[command('inspect_clue',clue_id='dispatch')])]:
            dump(root/f'{split}.json',[dict(id=split,split=split,exposure='development',family=TASK_ID,
                                          source=TASK_ID,prefix=prefix,anchor=command('query_status'))])
        dump(root/'reward_approval.json',approval)
        dump(root/'freeze.json',dict(task_id=TASK_ID,evaluation_scope=scope,
             files={n:digest(root/n) for n in ('train.json','eval.json','reward_approval.json')}))

    def test_development_is_not_holdout_and_freeze_is_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_suite(root)
            self.assertEqual(len(load_training_suite(root)[0]),1)
            path=root/'freeze.json';obj=read(path);obj['evaluation_scope']='independent_holdout'
            path.write_text(json.dumps(obj),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'relabel'):
                load_training_suite(root)
            (root/'train.json').write_text('[]',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'changed'):
                load_training_suite(root)

    def test_pending_calibration_rejected(self):
        with patch('characore.grpo_data.summarize',return_value={'calibration_passed':False}):
            with self.assertRaisesRegex(ValueError,'not passed'):
                verify_calibration(None,None,None)


if __name__=='__main__':
    unittest.main()
