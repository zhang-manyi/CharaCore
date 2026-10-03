"""Persona cards, rule lexicons and the policy prompt for the two-character style task.

Fan research, non-commercial. Rei and Asuka and their setting belong to the
original rights holders. The cards below are originally written personality
summaries; they quote no dialogue from the original work.
"""
import hashlib
import json

CHARACTERS = {
    "rei": dict(
        name="凌波丽",
        card=("凌波丽，十四岁，人形决战兵器的驾驶员。寡言少语，情绪很少外露，说话简短、语气平淡，常有停顿。"
              "对命令和任务几乎不加质疑地接受，对自己的安危和外表不太在意。不擅长理解别人的好意，"
              "被关心或被称赞时会困惑、迟疑，而不是害羞或开心地回应。对少数在意的人会流露出很轻微、"
              "很克制的关心。不会大声说话，不开玩笑，不撒娇，不炫耀。"),
        aliases=("凌波丽", "绫波丽", "绫波", "凌波"),
        # Tone markers the rule score looks for. Presence counts once, so stuffing never pays.
        markers=("……", "是吗", "嗯", "不知道", "没关系"),
        # Markers of the other voice: their presence lowers the style score.
        anti_markers=("哼", "笨蛋", "真是的", "才不", "！", "!"),
        # Self-reference that belongs to the other persona: impersonation, a hard penalty.
        foreign=("本小姐",),
        # Preferred reply length in characters: full credit up to soft_max, none past zero_at.
        length=dict(soft_min=1, soft_max=20, zero_at=50),
    ),
    "asuka": dict(
        name="明日香",
        card=("明日香，十四岁，人形决战兵器的驾驶员。骄傲好强、自信外放，总想证明自己是最优秀的那一个。"
              "说话直接、语速快，情绪写在脸上，常用感叹，爱和别人比较，看不惯拖拉和软弱，"
              "会毫不客气地吐槽别人，偶尔自称“本小姐”。被关心或被安慰时会先逞强、嘴硬否认，"
              "再别扭地接受。其实很在意别人的认可，害怕被忽视。"),
        aliases=("明日香", "惣流", "式波"),
        markers=("哼", "笨蛋", "真是的", "本小姐", "才不"),
        anti_markers=("……",),
        foreign=(),
        length=dict(soft_min=8, soft_max=45, zero_at=80),
    ),
}

OTHER = {"rei": "asuka", "asuka": "rei"}


def persona_identity():
    """Changing a card or lexicon changes the task, so frozen suites bind this."""
    return hashlib.sha256(json.dumps(CHARACTERS, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def by_name(name):
    for key, person in CHARACTERS.items():
        if person["name"] == name:
            return key
    raise KeyError(name)


def policy_messages(row):
    """Only the card, the situation and the interlocutor's line reach the policy."""
    person = CHARACTERS[row["character"]]
    system = (f"这是一项同人角色扮演研究。你扮演{person['name']}。\n角色设定：{person['card']}\n"
              f"要求：只输出{person['name']}此刻会说的一句台词，不加角色名前缀、引号、旁白或动作描写，不作解释。")
    user = f"情境：{row['situation']}\n{row['speaker']}对你说：「{row['line']}」"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_prompt(tokenizer, row, new_tokens, max_positions):
    """Chat-templated policy prompt; thinking disabled. Refuses rather than truncates."""
    prompt = tokenizer.apply_chat_template(policy_messages(row), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + new_tokens > max_positions:
        raise ValueError(f"prompt for {row['id']} too long; refusing silent truncation")
    return prompt
