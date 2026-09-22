"""Local, immutable judge calls using the existing v0.3 request/response contract."""
import hashlib
import json
from pathlib import Path

from characore.agent import dump
from characore.judge import parse_response


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def judge_identity(metadata):
    # Paths and purpose labels are not model identity. Weights, decoding and code are.
    keys = ("model_files_sha256", "packages", "source_sha256", "device", "dtype",
            "quantized_4bit", "seed", "do_sample", "max_new_tokens", "enable_thinking")
    return identity({k: metadata[k] for k in keys})


def call_judge(request, policy, output, retries=1, input_limit=4096):
    if type(retries) is not int or not 0 <= retries <= 1 or input_limit < 1:
        raise ValueError("at most one retry and a positive input budget required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "request.json", request)
    attempts = []
    required = json.loads(request["messages"][1]["content"])["action_required"]
    for number in range(retries + 1):
        try:
            rendered = policy.tokenizer.apply_chat_template(request["messages"], tokenize=False,
                                                            add_generation_prompt=True, enable_thinking=False)
            token_count = len(policy.tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if token_count > input_limit:
                raise ValueError(f"input budget exceeded: {token_count} > {input_limit}; no truncation")
            raw, usage = policy(request["messages"])
            result = parse_response(raw, request["allowed_evidence_ids"], required)
            result["usage"] = usage
        except Exception as exc:
            result = dict(call_status="inference_error", judgement=None, raw=None,
                          error=f"{type(exc).__name__}: {exc}")
        attempts.append(result)
        dump(output / f"attempt_{number + 1}.json", result)
        if result["call_status"] == "ok":
            break
    record = dict(id=request["id"], request_sha256=identity(request), attempts=attempts,
                  final=attempts[-1], judge_identity=judge_identity(policy.metadata))
    dump(output / "result.json", record)
    return record
