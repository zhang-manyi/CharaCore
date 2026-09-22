"""Evaluate actual human/judge results. Design expectations are never gold labels."""
from collections import Counter
import json
from pathlib import Path

from characore.calibration import verify_packet
from characore.judge import PROTOCOL, validate_response, parse_response
from characore.judge_runner import identity
from characore.protocol import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ratio(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                value=numerator / denominator if denominator else None)


def summarize(packet, results=None, human=None):
    packet = Path(packet)
    manifest = verify_packet(packet)
    requests = read(packet / "judge/requests.json")
    mapping = read(packet / "audit/mapping_and_expectations.json")
    blind = {r["id"]: r["material"] for r in read(packet / "human/review.json")["cases"]}
    request_by_id = {r["id"]: r for r in requests}
    finals, first_valid, attempts, fingerprints = {}, 0, 0, set()
    if results:
        for file in Path(results).glob("*/result.json"):
            record = read(file)
            rid = record["id"]
            if rid not in request_by_id or rid in finals:
                raise ValueError("unknown or duplicate judge request")
            request = request_by_id[rid]
            if record["request_sha256"] != identity(request):
                raise ValueError("judge request binding mismatch")
            if not 1 <= len(record["attempts"]) <= 2 or record["final"] != record["attempts"][-1]:
                raise ValueError("invalid retry history")
            required = json.loads(request["messages"][1]["content"])["action_required"]
            for attempt in record["attempts"]:
                if attempt["call_status"] == "ok":
                    parsed = parse_response(attempt["raw"], request["allowed_evidence_ids"], required)
                    if parsed["call_status"] != "ok" or parsed["judgement"] != attempt["judgement"]:
                        raise ValueError("judge raw response mismatch")
            first_valid += record["attempts"][0]["call_status"] == "ok"
            attempts += len(record["attempts"])
            fingerprints.add(record["judge_identity"])
            finals[rid] = record["final"]
    if len(fingerprints) > 1:
        raise ValueError("mixed judge identities")

    humans, reviewer_labels, human_hashes, citation_errors = {}, [], {}, None
    if human:
        human = Path(human)
        reviewers = []
        for filename in ("reviewer_1.json", "reviewer_2.json"):
            form = read(human / filename)
            if form["protocol"] != PROTOCOL or not form.get("reviewer") or not form.get("completed_at"):
                raise ValueError("human review is not signed and completed")
            reviewers.append(form["reviewer"])
            annotations = {r["id"]: r for r in form["annotations"]}
            if len(annotations) != len(form["annotations"]) or set(annotations) != set(blind):
                raise ValueError("human case set mismatch")
            labels = {}
            for bid, ann in annotations.items():
                if ann["review_status"] != "completed":
                    raise ValueError("human first round incomplete")
                obj = {k: ann[k] for k in ("winner", "reason", "preference_evidence_ids", "scores")}
                obj["protocol"] = PROTOCOL
                material = blind[bid]
                validate_response(obj, [e["id"] for e in material["evidence"]] + ["candidate:A", "candidate:B"],
                                  material["action_required"])
                labels[bid] = ann["winner"]
            reviewer_labels.append(labels)
            human_hashes[filename] = digest(human / filename)
        if reviewers[0] == reviewers[1]:
            raise ValueError("two distinct human reviewers required for this protocol")
        adjudication = read(human / "adjudication.json")
        if adjudication.get("reviewer_files_sha256") != human_hashes:
            raise ValueError("adjudication must bind locked first-round files")
        if not adjudication.get("reviewer") or not adjudication.get("completed_at"):
            raise ValueError("unsigned adjudication")
        annotations = {r["id"]: r for r in adjudication["annotations"]}
        if len(annotations) != len(adjudication["annotations"]) or set(annotations) != set(blind):
            raise ValueError("adjudication case set mismatch")
        citation_errors = 0
        for bid, ann in annotations.items():
            if ann["review_status"] != "completed":
                continue
            if ann["winner"] not in ("A", "B", "tie", "insufficient") or not ann.get("reason"):
                raise ValueError("invalid adjudication")
            refs = ann.get("preference_evidence_ids")
            allowed = {e["id"] for e in blind[bid]["evidence"]} | {"candidate:A", "candidate:B"}
            if not isinstance(refs, list) or any(r not in allowed for r in refs):
                raise ValueError("invalid adjudication citations")
            if ann["winner"] in ("A", "B") and not refs:
                raise ValueError("strict human winner needs evidence")
            winner = ann["winner"]
            humans[mapping[bid]["case"]["id"]] = mapping[bid]["display_to_original"].get(winner, winner)
        # Humans must actually inspect model reasons, not only label candidate pairs.
        audit = adjudication.get("model_citation_audit", {})
        if (audit.get("judge_identity") not in fingerprints or
                audit.get("reviewed_request_ids") != sorted(finals) or
                audit.get("results_sha256") != identity(finals) or
                len(finals) != len(requests) or audit.get("completed") is not True):
            citation_errors = None
        else:
            citation_errors = audit.get("unsupported_citations")
            if type(citation_errors) is not int or citation_errors < 0:
                raise ValueError("invalid human citation audit count")
        human_hashes["adjudication.json"] = digest(human / "adjudication.json")

    pairs, display_choices, states = {}, Counter(), Counter()
    for bid, entry in mapping.items():
        cid = entry["case"]["id"]
        labels = []
        for suffix in ("_AB", "_BA"):
            rid = cid + suffix
            result = finals.get(rid)
            if result and result["call_status"] == "ok":
                judgement = result["judgement"]
                winner = judgement["winner"]
                display_choices[winner] += 1
                for side, dims in judgement["scores"].items():
                    for dim, score in dims.items():
                        states[f"{suffix[1:]}:{side}:{dim}:{score['status']}"] += 1
                labels.append(request_by_id[rid]["display_to_original"].get(winner, winner))
        both = len(labels) == 2
        consistent = both and labels[0] == labels[1]
        pairs[cid] = dict(group=entry["split_group"], focus=entry["case"]["focus"],
                          both_valid=both, order_consistent=consistent,
                          winner=labels[0] if consistent else None, human=humans.get(cid),
                          agrees=consistent and cid in humans and labels[0] == humans[cid])
    total, valid = len(requests), sum(r["call_status"] == "ok" for r in finals.values())
    both = sum(p["both_valid"] for p in pairs.values())
    same = sum(p["order_consistent"] for p in pairs.values())
    decided = [p for p in pairs.values() if p["human"] in ("A", "B", "tie")]
    strict = [p for p in pairs.values() if p["human"] in ("A", "B")]
    insufficient = [p for p in pairs.values() if p["human"] == "insufficient"]
    all_agree = sum(p["agrees"] for p in pairs.values())
    gates = dict(all_calls_executed=len(finals) == total,
                 schema=valid / total >= .95,
                 enough_bidirectional=both >= len(pairs) * 33 / 36,
                 order=both > 0 and same / both >= .9,
                 all_human_adjudicated=len(humans) == len(pairs),
                 human_all=len(humans) > 0 and all_agree / len(humans) >= .8,
                 human_decidable=bool(decided) and sum(p["agrees"] for p in decided) / len(decided) >= .8,
                 human_citation_audit=citation_errors == 0)
    human_decidable_ids = ([bid for bid in blind if all(labels[bid] in ("A", "B", "tie")
                                                       for labels in reviewer_labels)]
                          if reviewer_labels else [])
    def by(field):
        return {group: dict(pairs=sum(p[field] == group for p in pairs.values()),
                            order_consistent=sum(p[field] == group and p["order_consistent"] for p in pairs.values()),
                            human_agree=sum(p[field] == group and p["agrees"] for p in pairs.values()))
                for group in sorted({p[field] for p in pairs.values()})}
    return dict(status="passed" if all(gates.values()) else "pending_or_failed", calibration_passed=all(gates.values()),
                scope=manifest.get("calibration_scope", "genshin-stage-b-dev-v0.3"),
                transferable_to_agent_task=False,
                packet_sha256=digest(packet / "manifest.json"), judge_identity=next(iter(fingerprints), None),
                judge_results_sha256=identity(finals), human_files_sha256=human_hashes,
                planned_calls=total, executed_calls=len(finals), uncalled=total-len(finals), attempts=attempts,
                first_valid=ratio(first_valid, total) if finals else dict(numerator=0, denominator=total, value=None),
                final_valid=ratio(valid, total) if finals else dict(numerator=0, denominator=total, value=None),
                order_consistency=ratio(same, both), order_consistency_all=ratio(same, len(pairs)),
                human_agreement=ratio(all_agree, len(humans)),
                human_decidable=ratio(sum(p["agrees"] for p in decided), len(decided)),
                human_strict=ratio(sum(p["agrees"] for p in strict), len(strict)),
                human_insufficient=ratio(sum(p["agrees"] for p in insufficient), len(insufficient)),
                inter_human=ratio(sum(reviewer_labels[0][b] == reviewer_labels[1][b] for b in blind), len(blind))
                            if reviewer_labels else ratio(0, 0),
                inter_human_decidable=ratio(sum(reviewer_labels[0][b] == reviewer_labels[1][b]
                                                for b in human_decidable_ids), len(human_decidable_ids)),
                unsupported_citations=citation_errors, gates=gates, pairs=pairs, groups=by("group"),
                diagnostics=by("focus"), displayed_winners=dict(display_choices), score_states=dict(states),
                claim="Development calibration only; uncalled is not measured failure; expectations are not gold.")
