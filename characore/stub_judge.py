"""Deterministic offline judge for wiring checks. NOT a quality signal.

Decides by the project's own rule style score, so it exercises the full request
build, strict parse, AB/BA order check, in-batch reuse and group rejection
without any network. A run using this backend proves the pipeline works; it
says nothing about whether replies are in character or whether the policy improved.
"""
import json

from characore.judge import ATTR_PROTOCOL, PROTOCOL
from characore.persona import CHARACTERS, by_name
from characore.protocol import digest
from characore.style_rewards import style_score


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
        self.calls += 1
        data = json.loads(messages[1]["content"])
        if data["protocol"] == ATTR_PROTOCOL:
            body = self.attribute(data)
        else:
            body = self.compare(data)
        return json.dumps(body, ensure_ascii=False), dict(finish_reason="stop", request_number=self.calls,
                                                          usage=None)

    @staticmethod
    def compare(data):
        """Prefer the displayed candidate with the higher rule style score; equal scores tie."""
        character = by_name(data["character"])
        score = {side: style_score(data[f"candidate_{side}"], character) for side in ("A", "B")}
        winner = "tie" if score["A"] == score["B"] else max(score, key=score.get)
        scores = {side: {dim: dict(status="scored", score=round(4 * score[side]) if dim == "P" else 3,
                                   reason="stub: rule style score", evidence_ids=["E1", f"candidate:{side}"])
                         for dim in "PRN"} for side in ("A", "B")}
        return dict(protocol=PROTOCOL, winner=winner, scores=scores,
                    reason="stub judge compares the rule style score only",
                    preference_evidence_ids=[] if winner == "tie" else ["E1", f"candidate:{winner}"])

    @staticmethod
    def attribute(data):
        """The persona whose tone markers appear more often; neither on a tie."""
        hits = {k: sum(m in data["reply"] for m in CHARACTERS[k]["markers"]) for k in ("rei", "asuka")}
        speaker = "neither" if hits["rei"] == hits["asuka"] else max(hits, key=hits.get)
        return dict(protocol=ATTR_PROTOCOL, speaker=speaker, reason="stub: tone marker count")
