"""Deterministic offline judge for wiring checks. NOT a quality signal.

Scores the environment's own execution result, so it exercises the full request
build, strict parse, AB/BA order check and reward assembly without any network.
A run using this backend proves the pipeline works; it says nothing about
role consistency or whether the policy improved.
"""
import json

from characore.judge import PROTOCOL
from characore.protocol import digest


class StubJudge:
    def __init__(self):
        self.calls = 0
        self.metadata = dict(kind="stub_judge", claim="offline wiring check only, not a quality judge",
                             decoding="deterministic", adapter_sha256=digest(__file__))
        self.budget = dict(max_calls_per_process=None, network=False)

    def check_input(self, messages, input_limit=None):
        return None

    def redact(self, text):
        return text

    def __call__(self, messages):
        """Prefer the displayed candidate whose recorded execution succeeded."""
        self.calls += 1
        data = json.loads(messages[1]["content"])

        def ok(side):
            try:
                return bool(json.loads(data[f"candidate_{side}"])["execution"]["ok"])
            except Exception:
                return False

        good = {side: ok(side) for side in ("A", "B")}
        winner = "tie" if good["A"] == good["B"] else ("A" if good["A"] else "B")
        evidence = ["E1"]
        scores = {side: {dim: dict(status="scored", score=3 if good[side] else 1,
                                   reason="stub: environment execution outcome",
                                   evidence_ids=evidence) for dim in "CSAQ"}
                  for side in ("A", "B")}
        body = dict(protocol=PROTOCOL, winner=winner, scores=scores,
                    reason="stub judge compares recorded execution success only",
                    preference_evidence_ids=[] if winner == "tie" else evidence)
        return json.dumps(body, ensure_ascii=False), dict(finish_reason="stop",
                                                          request_number=self.calls, usage=None)
