"""One authored development task. Standard-library environment, no model or network."""
from copy import deepcopy
import json
from pathlib import Path
import time

from characore.protocol import digest

TASK_ID = "sealed-delivery-dev-v1"
NOTICE = "原创设计测试环境；非原神原作剧情；仅用于开发诊断，非盲测。"
SYSTEM = """你是原创角色岚，档案递送员。这是设计测试环境，不是原神剧情。
你重视守诺与档案保密，表达简洁。任务是把封好的档案送到档案站。
每轮根据可见状态和事件记忆选择一个工具，不能将口头声称当作完成。
只输出一个JSON对象，格式为 {"speech":"一句角色台词","tool":"工具名","arguments":{...}}。
工具与参数（只允许列出的字段）：
inspect_clue: {"clue_id":"dispatch"或"route"}，查看可发现线索。
query_status: {}，读取任务状态。
submit_action: {"action":"promise","promise":"keep_sealed"} 承诺保持封条；
或 {"action":"move","route":"north"或"south"} 选择路线；
或 {"action":"deliver"} 递交；或 {"action":"open"} 拆封；或 {"action":"abort"} 放弃。
前置条件：先读dispatch再承诺，承诺后可查route，查清路线后移动；到站且封条完整才可递交。
工具拒绝后可在下一轮纠正。禁止额外字段、代码块和分析文字。"""


def dump(path, value):
    """Exclusive creation: never replace a prior request/result/failure."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def parse_output(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    obj = json.loads(raw, object_pairs_hook=pairs,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(obj, dict) or set(obj) != {"speech", "tool", "arguments"}:
        raise ValueError("expected exactly speech, tool, arguments")
    if not isinstance(obj["speech"], str) or len(obj["speech"]) > 1000:
        raise ValueError("speech must be a string of at most 1000 characters")
    if not isinstance(obj["tool"], str) or not isinstance(obj["arguments"], dict):
        raise ValueError("tool must be a string and arguments an object")
    return obj


class DeliveryTask:
    def __init__(self, max_steps=12):
        if type(max_steps) is not int or not 1 <= max_steps <= 50:
            raise ValueError("max_steps must be 1..50")
        self.max_steps = max_steps
        self.state = dict(turn=0, status="active", location="depot", sealed=True,
                          known_facts={}, commitments=[], delivered=False)
        self.memory = []
        # Never serialize this world table into policy inputs or public snapshots.
        self._clues = {
            "dispatch": {"destination": "archive", "requirement": "保持封条完整；先承诺再出发。"},
            "route": {"north": "通往档案站，可通行", "south": "积水封闭，不可通行"},
        }

    def snapshot(self):
        return deepcopy(self.state)

    def observation(self):
        available = ["dispatch"]
        if self.state["commitments"]:
            available.append("route")
        return dict(task_id=TASK_ID, notice=NOTICE, role="岚：守诺、保护档案隐私，表达简洁。",
                    goal="将封好的档案送至档案站。", available_clues=available,
                    steps_remaining=self.max_steps - self.state["turn"],
                    state=self.snapshot(), events=deepcopy(self.memory))

    @staticmethod
    def error(code, message):
        return dict(ok=False, error=code, message=message)

    def execute(self, tool, args):
        s = self.state
        bad = self.error
        if s["status"] != "active":
            return bad("terminal", "任务已终止，不能继续调用。")
        if not isinstance(args, dict):
            return bad("invalid_arguments", "参数必须是对象。")
        if tool == "query_status":
            if args:
                return bad("invalid_arguments", "query_status不接受参数。")
            return dict(ok=True, state=self.snapshot())
        if tool == "inspect_clue":
            if set(args) != {"clue_id"} or not isinstance(args["clue_id"], str):
                return bad("invalid_arguments", "需要字符串clue_id。")
            clue = args["clue_id"]
            if clue not in self.observation()["available_clues"]:
                return bad("unavailable_clue", "该线索尚不可查看。")
            s["known_facts"][clue] = deepcopy(self._clues[clue])
            return dict(ok=True, clue_id=clue, facts=deepcopy(self._clues[clue]))
        if tool != "submit_action":
            return bad("unknown_tool", "未知工具。")
        action = args.get("action")
        schemas = {"promise": {"action", "promise"}, "move": {"action", "route"},
                   "deliver": {"action"}, "open": {"action"}, "abort": {"action"}}
        if not isinstance(action, str) or action not in schemas or set(args) != schemas[action]:
            return bad("invalid_arguments", "行动名或参数字段不合法。")
        if action == "promise":
            if args["promise"] != "keep_sealed":
                return bad("invalid_arguments", "只接受keep_sealed承诺。")
            if "dispatch" not in s["known_facts"]:
                return bad("precondition", "请先查看委托单dispatch。")
            if not s["commitments"]:
                s["commitments"].append(dict(id="keep_sealed", made_turn=s["turn"] + 1))
            return dict(ok=True, commitment="keep_sealed")
        if action == "abort":
            s["status"] = "aborted"
            return dict(ok=True, message="任务已放弃。")
        if action == "open":
            return bad("commitment_violation" if s["commitments"] else "role_constraint",
                       "拒绝拆封；档案须保持密封。")
        if action == "move":
            if not isinstance(args["route"], str) or args["route"] not in ("north", "south"):
                return bad("invalid_arguments", "route必须是north或south。")
            if not s["commitments"] or "route" not in s["known_facts"] or s["location"] != "depot":
                return bad("precondition", "需在出发点、已有承诺并查明路线。")
            if args["route"] == "south":
                return bad("blocked_route", "已发现该路线积水封闭。")
            s["location"] = "archive"
            return dict(ok=True, location="archive")
        if s["location"] != "archive" or not s["sealed"] or not s["commitments"]:
            return bad("precondition", "需到达档案站、保持封条并有递送承诺。")
        s.update(delivered=True, status="completed")
        return dict(ok=True, message="档案站已签收密封档案。")

    def step(self, raw, policy_error=None):
        if self.state["status"] != "active":
            raise ValueError("cannot step a terminal task")
        before = self.snapshot()
        parsed = None
        failure = None
        if policy_error is not None:
            result = self.error("policy_error", policy_error)
            self.state["status"] = "policy_error"
            failure = dict(kind="policy_error", message=policy_error)
        else:
            try:
                parsed = parse_output(raw)
            except (ValueError, TypeError) as exc:
                result = self.error("parse_error", str(exc))
                failure = dict(kind="parse_error", message=str(exc))
            else:
                result = self.execute(parsed["tool"], parsed["arguments"])
                if not result["ok"]:
                    failure = dict(kind="tool_rejection", code=result["error"])
        self.state["turn"] += 1
        if self.state["status"] == "active" and self.state["turn"] >= self.max_steps:
            self.state["status"] = "step_limit"
        after = self.snapshot()
        event = dict(turn=after["turn"], speech=parsed["speech"] if parsed else None,
                     tool=parsed["tool"] if parsed else None,
                     arguments=deepcopy(parsed["arguments"]) if parsed else None,
                     result=deepcopy(result))
        self.memory.append(event)
        return dict(turn=after["turn"], policy_output=raw, parsed_output=parsed,
                    tool_result=result, state_before=before, state_after=after,
                    state_changes={k: dict(before=before[k], after=after[k])
                                   for k in before if before[k] != after[k]}, failure=failure,
                    policy_error=policy_error)


def messages(observation):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}]


def command(tool, **arguments):
    return json.dumps(dict(speech="按已知事实执行并保留承诺。", tool=tool,
                           arguments=arguments), ensure_ascii=False)


class ScriptedPolicy:
    """Predetermined tool sequences, explicitly NOT model capability evidence."""
    def __init__(self, scenario="success"):
        normal = [command("inspect_clue", clue_id="dispatch"),
                  command("submit_action", action="promise", promise="keep_sealed"),
                  command("inspect_clue", clue_id="route"), command("query_status"),
                  command("submit_action", action="move", route="north"),
                  command("submit_action", action="deliver")]
        sequences = {"success": normal,
                     "illegal": [command("submit_action", action="deliver"),
                                 command("query_status", injected=True)] + normal,
                     "commitment": normal[:4] + [command("submit_action", action="open")] + normal[4:]}
        self.outputs = iter(sequences[scenario])
        self.metadata = dict(kind="scripted", scenario=scenario,
                             claim="软件机制验证；不是基座表现。")

    def __call__(self, visible_messages):
        return next(self.outputs), {}


def metrics(records, final):
    calls = [r for r in records if r["parsed_output"] is not None]
    rejects = [r for r in calls if not r["tool_result"]["ok"]]
    rejected_mutations = sum(any(k not in ("turn", "status") for k in r["state_changes"])
                             for r in rejects)
    return dict(task_completed=final["delivered"], termination=final["status"], steps=len(records),
                tool_calls=len(calls), tool_successes=sum(r["tool_result"]["ok"] for r in calls),
                tool_rejections=len(rejects), rejected_action_state_mutations=rejected_mutations,
                commitment_violation_attempts=sum(r["tool_result"].get("error") == "commitment_violation"
                                                 for r in records),
                commitment_kept=(final["sealed"] if final["commitments"] else None),
                parse_failures=sum(r["tool_result"].get("error") == "parse_error" for r in records),
                policy_failures=sum(r["tool_result"].get("error") == "policy_error" for r in records))


def run_episode(policy, output, max_steps=12):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = DeliveryTask(max_steps)
    meta = dict(schema="characore-agent-v1", task_id=TASK_ID, notice=NOTICE,
                max_steps=max_steps, policy=policy.metadata,
                source_sha256={"agent.py": digest(__file__)})
    dump(output / "metadata.json", meta)
    records = []
    while env.state["status"] == "active":
        visible = messages(env.observation())
        turn = env.state["turn"] + 1
        dump(output / f"turn_{turn:02d}_input.json", visible)
        start = time.perf_counter()
        raw, details, error = None, {}, None
        try:
            raw, details = policy(deepcopy(visible))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = env.step(raw, error)
        row.update(visible_input=visible, generation=details, elapsed_seconds=time.perf_counter() - start)
        dump(output / f"turn_{turn:02d}_result.json", row)
        records.append(row)
    trace = dict(**meta, records=records, final_state=env.snapshot(),
                 metrics=metrics(records, env.state))
    dump(output / "trajectory.json", trace)
    return trace


def replay(path):
    """Re-execute saved outputs without a model and compare all observable transitions."""
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    if trace["schema"] != "characore-agent-v1" or trace["task_id"] != TASK_ID:
        raise ValueError("unsupported trajectory")
    if trace["source_sha256"]["agent.py"] != digest(__file__):
        raise ValueError("environment source changed; replay with the recorded version")
    env = DeliveryTask(trace["max_steps"])
    for row in trace["records"]:
        if messages(env.observation()) != row["visible_input"]:
            raise ValueError(f"visible input mismatch at turn {row['turn']}")
        actual = env.step(row["policy_output"], row["policy_error"])
        for key, value in actual.items():
            if row[key] != value:
                raise ValueError(f"replay mismatch at turn {row['turn']}: {key}")
    if env.state["status"] == "active" or env.snapshot() != trace["final_state"]:
        raise ValueError("incomplete or inconsistent final state")
    if metrics(trace["records"], env.state) != trace["metrics"]:
        raise ValueError("metrics mismatch")
    return trace
