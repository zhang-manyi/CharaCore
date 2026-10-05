"""Check .env locally; --allow-api sends exactly one synthetic judge request, no retry."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from characore.api_judge import APIJudge
from characore.judge import make_request
from characore.judge_runner import call_judge
from characore.protocol import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.allow_api and args.output is None:
        parser.error("--allow-api requires a new --output directory")
    policy = APIJudge(args.env_file, allow_calls=args.allow_api, backoff=())
    print(json.dumps(dict(config_valid=True, api_key_present=True, **policy.metadata,
                          budget=policy.budget, network_called=False), ensure_ascii=False))
    if not args.allow_api:
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "model.json", dict(metadata=policy.metadata, budget=policy.budget))
    context = dict(character="rei", situation="放学后的教室里只剩你们两个人。", speaker="同班同学",
                   line="你带伞了吗？")
    req = make_request(context, {"A": "……没有。", "B": "哼，本小姐才不需要伞！"})
    req["id"] = "api_smoke"
    result = call_judge(req, policy, args.output / "call", retries=0)
    print(json.dumps(dict(status=result["final"]["call_status"], calibration_passed=False,
                          claim="Connectivity/schema check only, not judge calibration")))
    return 0 if result["final"]["call_status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
