"""Run existing calibration requests locally or evaluate supplied human/model results."""
import argparse
import json
import re
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from characore.agent import dump
from characore.calibration import verify_packet
from characore.calibration_report import read, summarize
from characore.judge_runner import call_judge, judge_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "report"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--human", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, help="Smoke subset; uncalled requests remain uncalled")
    args = parser.parse_args()
    verify_packet(args.packet)
    if args.mode == "run" and not args.base:
        parser.error("run requires --base local directory")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        results = args.results
        if args.mode == "run":
            from characore.local_policy import LocalPolicy
            policy = LocalPolicy(args.base, device=args.device, max_new_tokens=1024)
            metadata = dict(policy.metadata, claim="Local candidate judge, reliability not established")
            dump(args.output / "model.json", metadata)
            requests = {r["id"]: r for r in read(args.packet / "judge/requests.json")}
            schedule = read(args.packet / "audit/request_schedule.json")["request_ids"]
            if len(set(schedule)) != len(schedule) or set(schedule) != set(requests):
                raise ValueError("calibration schedule must cover each request exactly once")
            if any(not re.fullmatch(r"[A-Za-z0-9_-]+", rid) for rid in schedule):
                raise ValueError("request IDs must be safe local filenames")
            results = args.output / "calls"
            results.mkdir()
            for rid in schedule[:args.limit]:
                result = call_judge(requests[rid], policy, results / rid)
                print(json.dumps(dict(id=rid, status=result["final"]["call_status"])), flush=True)
        report = summarize(args.packet, results, args.human)
        dump(args.output / "report.json", report)
        print(json.dumps({k: report[k] for k in ("calibration_passed", "executed_calls", "uncalled", "human_agreement")}))
        return 0
    except Exception:
        dump(args.output / "failure.json", dict(traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
