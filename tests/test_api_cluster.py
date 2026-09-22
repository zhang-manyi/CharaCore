"""Offline API transport fixtures and distributed reward/precision checks; no paid calls."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from characore.api_judge import APIJudge, NoRedirect, settings
from characore.distributed_rewards import check_payloads
from characore.grpo_rewards import UnusableReward
from characore.judge_runner import call_judge, judge_identity
from characore.precision import select_precision


class APIClusterTests(unittest.TestCase):
    @contextmanager
    def config(self, style="chat_completions", **extras):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / ".env"
            lines = dict(BASE_URL="https://api.openai.com/v1", API_KEY="synthetic-secret-only",
                         MODEL="gpt-6-astra", API_STYLE=style, **extras)
            path.write_text("\n".join(f"CHARACORE_JUDGE_{k}={v}" for k,v in lines.items()), encoding="utf-8")
            yield path

    def test_config_is_literal_and_never_calls_network(self):
        with self.config() as path, patch("urllib.request.build_opener") as network:
            policy = APIJudge(path)
            self.assertEqual(policy.model, "gpt-6-astra")
            self.assertEqual(policy.endpoint,"https://api.openai.com/v1/chat/completions")
            self.assertNotIn("synthetic-secret-only",json.dumps(policy.metadata))
            with self.assertRaisesRegex(ValueError, "disabled"):
                policy([{"role":"user","content":"test"}])
            network.assert_not_called()
            os.environ["CHARACORE_JUDGE_MODEL"] = "literal-$MODEL"
            self.assertEqual(APIJudge(path).model,"literal-$MODEL")

    def test_https_and_no_query_secrets(self):
        with self.config() as path:
            for value in ("http://example.com", "https://user:key@example.com", "https://example.com?key=x"):
                with patch.dict(os.environ, {"CHARACORE_JUDGE_BASE_URL":value}):
                    with self.assertRaises(ValueError):
                        APIJudge(path)
            self.assertIsNone(NoRedirect().redirect_request(None,None,302,None,None,"https://other.example"))

    def test_budget_blocks_before_transport(self):
        with self.config(MAX_CALLS="1", MAX_INPUT_BYTES="10") as path, patch("urllib.request.build_opener") as network:
            policy=APIJudge(path,allow_calls=True)
            with self.assertRaisesRegex(ValueError,"byte budget"):
                policy([{"role":"user","content":"test"}])
            policy.calls=1
            with self.assertRaisesRegex(ValueError,"budget exhausted"):
                policy([])
            network.assert_not_called()

    def test_chat_request_and_response_capture_redacts_secret(self):
        body=dict(model="gpt-6-astra",choices=[dict(message=dict(content='{"reason":"synthetic-secret-only"}'),finish_reason="stop")],usage={"total_tokens":12})
        with self.config() as path, patch("urllib.request.build_opener") as factory:
            factory.return_value.open.return_value=io.BytesIO(json.dumps(body).encode())
            policy=APIJudge(path,allow_calls=True)
            raw, usage=policy([{"role":"user","content":"JSON test"}])
            req=factory.return_value.open.call_args.args[0]
            payload=json.loads(req.data)
            self.assertEqual(payload["model"],"gpt-6-astra")
            self.assertEqual(payload["max_completion_tokens"],4096)
            self.assertNotIn("temperature",payload)
            self.assertFalse(payload["store"])
            self.assertNotIn("synthetic-secret-only",raw+json.dumps(usage))
            self.assertEqual(usage["usage"]["total_tokens"],12)

    def test_responses_extracts_only_assistant_text(self):
        body=dict(model="gpt-6-astra",status="completed",output=[
            dict(type="reasoning",summary=[]),
            dict(type="message",role="assistant",content=[dict(type="output_text",text='{"winner":"tie"}')])])
        with self.config(style="responses") as path, patch("urllib.request.build_opener") as factory:
            factory.return_value.open.return_value=io.BytesIO(json.dumps(body).encode())
            policy=APIJudge(path,allow_calls=True)
            raw, usage=policy([{"role":"user","content":"JSON test"}])
            payload=json.loads(factory.return_value.open.call_args.args[0].data)
            self.assertIn("input",payload)
            self.assertNotIn("messages",payload)
            self.assertEqual(raw,'{"winner":"tie"}')
            self.assertEqual(usage["finish_reason"],"completed")

    def test_http_failure_never_discloses_provider_body_or_key(self):
        with self.config() as path, patch("urllib.request.build_opener") as factory:
            factory.return_value.open.side_effect=HTTPError("https://example",401,"synthetic-secret-only",{},io.BytesIO(b"secret body"))
            with self.assertRaisesRegex(RuntimeError,"HTTP 401") as caught:
                APIJudge(path,allow_calls=True)([])
            self.assertNotIn("synthetic-secret-only",str(caught.exception))
            self.assertNotIn("secret body",str(caught.exception))
            self.assertEqual(caught.exception.response_text,"secret body")

    def test_incomplete_or_invalid_provider_response_is_preserved(self):
        from characore.judge import make_request
        context=dict(id="test", character="Guard", action_required=True,
                     visible_turns=[dict(source_line=1,speaker="Player",text="Check first")])
        req=make_request(context,{"A":"Check","B":"Open"},{})
        req["id"]="test"
        with self.config(style="responses") as path, patch("urllib.request.build_opener") as factory:
            factory.return_value.open.return_value=io.BytesIO(json.dumps(dict(status="incomplete",output=[])).encode())
            result=call_judge(req,APIJudge(path,allow_calls=True),path.parent/'incomplete',retries=0)
            self.assertEqual(result["final"]["call_status"],"incomplete_response")
            self.assertIsNone(result["final"]["judgement"])
            self.assertIn('incomplete',result["final"]["usage"]["provider_response_text"])
            factory.return_value.open.return_value=io.BytesIO(b"bad provider JSON synthetic-secret-only")
            result=call_judge(req,APIJudge(path,allow_calls=True),path.parent/'invalid',retries=0)
            self.assertEqual(result["final"]["call_status"],"inference_error")
            self.assertNotIn("synthetic-secret-only",json.dumps(result))
            self.assertIn("bad provider JSON",result["final"]["provider_response_text"])

    def test_env_duplicates_unknown_and_missing_fields(self):
        with self.config() as path:
            with path.open("a",encoding="utf8") as f:f.write("\nCHARACORE_JUDGE_MODEL=another")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                settings(path)
            path.write_text("CHARACORE_JUDGE_TYPO=1",encoding="utf8")
            with self.assertRaisesRegex(ValueError,"unknown"):
                settings(path)
            path.write_text("",encoding="utf8")
            with self.assertRaisesRegex(ValueError,"missing"):
                APIJudge(path)

    def test_identity_binds_endpoint_model_decoding_not_secret(self):
        with self.config() as path:
            first=judge_identity(APIJudge(path).metadata)
            with patch.dict(os.environ,{"CHARACORE_JUDGE_API_KEY":"new-test-key"}):
                self.assertEqual(first,judge_identity(APIJudge(path).metadata))
            with patch.dict(os.environ,{"CHARACORE_JUDGE_MODEL":"different"}):
                self.assertNotEqual(first,judge_identity(APIJudge(path).metadata))

    def test_v100_auto_uses_native_fp16(self):
        self.assertEqual(select_precision("cuda",capability=(7,0)),"fp16")
        self.assertEqual(select_precision("cuda:1",capability=(8,0)),"bf16")
        self.assertEqual(select_precision("cpu"),"fp32")
        with self.assertRaisesRegex(ValueError,"native BF16"):
            select_precision("cuda","bf16",(7,0))
        with self.assertRaises(ValueError):
            select_precision("cpu","fp16")

    def test_cross_rank_rewards_groups_and_failure(self):
        payloads=[dict(values=[0],keys=["same"],error=None),dict(values=[1],keys=["same"],error=None)]
        check_payloads(payloads,2)
        payloads[1]["keys"]=["different"]
        with self.assertRaisesRegex(UnusableReward,"different"):
            check_payloads(payloads,2)
        payloads[1]["error"]="TimeoutError"
        with self.assertRaisesRegex(UnusableReward,"all ranks"):
            check_payloads(payloads,2)
        payloads[1]=dict(values=[0],keys=["same"],error=None)
        with self.assertRaisesRegex(UnusableReward,"all-equal"):
            check_payloads(payloads,2)


if __name__ == "__main__":
    unittest.main()
