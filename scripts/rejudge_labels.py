"""Rejudge a labelled blind pair set with the current judge protocol and measure agreement with the labels.

Inputs are the label set's own files: label.html (the exact texts the labeller saw), key.json (which
side was which) and the exported labels. Each pair is judged AB/BA under the current cards and
protocol, both texts first losing an outer 「」 as in training, and scored as the reward would be:
the mean P+R+N margin of the first source (policy, or the opener reply) over the second.

Agreement is reported on the test and train pairs, as the v3 figures were; opener pairs (does the
judge still prefer a fixed opener?) are reported separately. The gate compares with v3 on label_01:
the margin's sign agreement on pairs where both human and judge were decisive, and the Pearson
correlation of the human score with the margin, must both be higher. The decision stays with the operator.

--canon adds the frozen original lines as evidence E4. Run once with and once without it, into two
output directories, to see whether the reference lines bring the judge closer to the labeller.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.canon import canon_identity, load_canon
from characore.judge import PROTOCOL
from characore.protocol import dump, read_json
from characore.style_data import load_suite
from characore.style_rewards import PairJudge, clean_reply, pair_outcome

VALUE = dict(L2=2, L1=1, EQ=0, R1=-1, R2=-2)
FIRST = dict(test="policy", train="policy", opener="opener")
# label_01 under style-v3 (test + train pairs): sign agreement on pairs both sides called, and Pearson.
V3_REFERENCE = dict(protocol="style-v3", verdict_decisive=(31, 42), margin_decisive=(41, 58),
                    pearson_margin=0.38, pearson_verdict=0.34)


def sign(x):
    return (x > 0) - (x < 0)


def page_items(path):
    """The item list embedded in label.html by make_label_set.py."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (\{.*?\});\s*\n", text, re.S)
    if not match:
        raise ValueError("label.html carries no embedded item data")
    return {item["id"]: item for item in json.loads(match.group(1).replace("<\\/", "</"))["items"]}


def agreement(items, judge_key):
    """Sign agreement (all pairs and decisive pairs) and Pearson of human score vs one judge quantity."""
    pairs = [(x["human"], x[judge_key]) for x in items]
    decisive = [(h, j) for h, j in pairs if h != 0 and sign(j) != 0]
    out = dict(pairs=len(pairs), exact_three_way=sum(sign(h) == sign(j) for h, j in pairs),
               decisive=len(decisive), decisive_agree=sum(sign(h) == sign(j) for h, j in decisive))
    out["decisive_rate"] = round(out["decisive_agree"] / len(decisive), 3) if decisive else None
    try:
        out["pearson"] = round(statistics.correlation([h for h, _ in pairs], [j for _, j in pairs]), 3)
    except statistics.StatisticsError:
        out["pearson"] = None  # fewer than two pairs, or a constant series
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True, help="holds label.html and key.json")
    parser.add_argument("--labels", type=Path, required=True, help="the exported labels JSON")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("api", "stub"), default="api")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--canon", type=Path, help="frozen canon reference set shown to the judge as evidence E4")
    args = parser.parse_args()
    if args.backend == "api" and not args.allow_api:
        parser.error("API judging requires --allow-api")
    canon = load_canon(args.canon) if args.canon else None
    train, test = load_suite(args.suite)
    rows = {r["id"]: r for r in train + test}
    key = read_json(args.label_dir / "key.json")
    shown = page_items(args.label_dir / "label.html")
    labels = read_json(args.labels)["labels"]
    if set(key["items"]) != set(shown):
        raise ValueError("key.json and label.html list different items")
    if args.backend == "api":
        from characore.api_judge import APIJudge
        judge = APIJudge(args.env_file, allow_calls=True)
    else:
        from characore.stub_judge import StubJudge
        judge = StubJudge()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = PairJudge(judge, args.output / "calls", args.workers, canon=canon)

    items, pending = [], {}
    for qid, k in sorted(key["items"].items()):
        texts = {k["left"]: shown[qid]["left"], k["right"]: shown[qid]["right"]}
        first = FIRST[k["kind"]]
        second = next(s for s in texts if s != first)
        choice = labels.get(qid, {}).get("choice")
        item = dict(qid=qid, kind=k["kind"], row_id=k["row_id"], character=rows[k["row_id"]]["character"],
                    first=clean_reply(texts[first]), second=clean_reply(texts[second]),
                    human=None if choice in (None, "SKIP") else VALUE[choice] * (1 if k["left"] == first else -1),
                    neither=bool(labels.get(qid, {}).get("neither")))
        for name in ("v3_pairwise", "v3_margin"):
            if name in k:
                item[name] = k[name]
        if item["first"] != item["second"]:
            reqs = pairs.requests(rows[k["row_id"]], item["first"], item["second"])
            item["keys"] = {label: key_ for label, (key_, _) in reqs.items()}
            for key_, request in reqs.values():
                pending.setdefault(key_, request)
        items.append(item)
    calls = pairs.run(pending) if pending else {}

    for item in items:
        if "keys" not in item:
            item.update(status="identical", margin=0.0, verdict=0.5)
            continue
        try:
            item.update(status="ok", **pair_outcome(calls[item["keys"]["AB"]], calls[item["keys"]["BA"]]))
        except ValueError as exc:
            item.update(status="unusable", reason=str(exc))

    usable = [x for x in items if x["status"] != "unusable" and x["human"] is not None]
    for x in usable:
        x["verdict_signed"] = x["verdict"] - 0.5
    judged = [x for x in usable if x["kind"] in ("test", "train")]
    openers = [x for x in usable if x["kind"] == "opener"]
    margin, verdict = agreement(judged, "margin"), agreement(judged, "verdict_signed")
    v3_rate = V3_REFERENCE["margin_decisive"][0] / V3_REFERENCE["margin_decisive"][1]
    beats = (margin["decisive_rate"] is not None and margin["decisive_rate"] > round(v3_rate, 3)
             and margin["pearson"] is not None and margin["pearson"] > V3_REFERENCE["pearson_margin"])
    by_character = {c: agreement([x for x in judged if x["character"] == c], "margin") for c in ("rei", "asuka")}
    summary = dict(protocol=PROTOCOL, canon_sha256=canon_identity(args.canon), label_set=key["set_id"], items=len(items),
                   labelled=sum(x["human"] is not None for x in items),
                   unusable=sum(x["status"] == "unusable" for x in items),
                   identical_after_cleaning=sum(x["status"] == "identical" for x in items),
                   order_inconsistent=sum(bool(x.get("order_inconsistent")) for x in items),
                   margin=margin, verdict=verdict, margin_by_character=by_character,
                   openers=dict(pairs=len(openers),
                                judge_prefers_opener=sum(x["margin"] > 0 for x in openers),
                                judge_prefers_without=sum(x["margin"] < 0 for x in openers),
                                human_prefers_opener=sum(x["human"] > 0 for x in openers),
                                human_prefers_without=sum(x["human"] < 0 for x in openers),
                                agreement=agreement(openers, "margin")),
                   v3_reference=V3_REFERENCE,
                   gate=dict(rule="margin decisive agreement rate and Pearson both above v3", passed=beats),
                   judge=judge.metadata, calls_made=judge.calls,
                   claim="one labeller, 100 pairs; agreement with that labeller only")
    dump(args.output / "items.json", items)
    dump(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("labelled", "unusable", "margin", "verdict", "openers", "gate")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
