"""Build and freeze the two-character style suite from original situation templates.

Every situation and line below is written for this project; none is quoted from
the original work. Each template is used for both characters, and train/test are
split by template so test situations are never seen in training.
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.persona import CHARACTERS, persona_identity
from characore.protocol import digest, dump
from characore.style_data import load_suite

TEST_TEMPLATES = 6
SEED = 17

# (template id, situation, interlocutor, five lines)
TEMPLATES = [
    ("T01", "放学后的教室里只剩你们两个人，窗外在下雨。", "同班同学",
     ["你带伞了吗？", "要不要一起走到车站？", "今天的小测你考得怎么样？", "你平时放学后都做什么？", "这雨好像一时停不了。"]),
    ("T02", "同步测试刚结束，你从测试舱里出来。", "技术主管",
     ["这次的数值比上次低了一些。", "身体有没有哪里不舒服？", "下周要加一轮测试。", "你觉得刚才哪里出了问题？", "先去休息吧，记录我来整理。"]),
    ("T03", "午休时，有人在食堂把便当递到你面前。", "同队的少年驾驶员",
     ["这是我多做的一份，你要吗？", "里面没有放肉。", "合不合你的口味？", "明天还想吃吗？", "你一个人吃饭不觉得无聊吗？"]),
    ("T04", "你在走廊被叫住，对方手里拿着你的成绩单。", "班主任",
     ["这次的成绩下滑得很厉害。", "训练和学业兼顾很辛苦吧？", "有什么困难可以跟我说。", "家长会谁会来参加？", "下次考试有信心吗？"]),
    ("T05", "出击前的待机室，警报灯在闪。", "同队的少年驾驶员",
     ["我有点害怕。", "这次的敌人好像很强。", "如果我拖了后腿怎么办？", "你一点都不紧张吗？", "活着回来吧。"]),
    ("T06", "战斗结束后，你躺在医院的病床上，有人来探望。", "同队的少年驾驶员",
     ["你醒了？伤口还疼吗？", "我带了些水果来。", "医生说你要住一周。", "那时候你为什么要挡在我前面？", "需要我帮你拿什么吗？"]),
    ("T07", "周末的商场里，你偶然碰到熟人。", "同班的女同学",
     ["好巧，你也来买东西？", "这条裙子很适合你哦。", "要不要一起去喝杯饮料？", "你平时穿便服的样子很少见呢。", "下次一起来逛吧。"]),
    ("T08", "训练场上，你的射击成绩刚刚公布。", "作战指挥官",
     ["命中率是全队第一。", "反应速度还可以再快一点。", "保持这个状态。", "你对这个结果满意吗？", "明天开始换新的训练科目。"]),
    ("T09", "深夜的宿舍楼下，有人看到你独自坐在台阶上。", "同队的少年驾驶员",
     ["这么晚了还不睡吗？", "你在看星星？", "我也睡不着。", "明天还有训练呢。", "我可以坐在这里吗？"]),
    ("T10", "生日那天，有人把一个小盒子放在你桌上。", "同队的少年驾驶员",
     ["生日快乐。", "不知道你喜欢什么，就随便挑了一个。", "打开看看吧。", "不喜欢的话我可以去换。", "你以前怎么过生日？"]),
    ("T11", "作战会议上，指挥官刚刚宣布由你担任主攻。", "作战指挥官",
     ["这次的主攻交给你。", "有没有问题？", "风险很高，你可以拒绝。", "其他人负责掩护你。", "我相信你能做到。"]),
    ("T12", "你在实验室外的长椅上等待，有人走过来搭话。", "研究所的研究员",
     ["今天的检查要花很长时间。", "你好像总是一个人等在这里。", "要喝点热的吗？", "检查的时候会有点冷。", "你对这些检查有什么想法？"]),
    ("T13", "放学路上，有男生在路口拦住你。", "隔壁班的男生",
     ["请问能和你交个朋友吗？", "我一直很在意你。", "能告诉我你的联系方式吗？", "周末有空一起看电影吗？", "抱歉，突然说这些。"]),
    ("T14", "教室里，你的作业本被同学捡起来看了一眼。", "同班同学",
     ["你的字写得好工整。", "这道题你是怎么做出来的？", "能借我抄一下吗？", "你连附加题都做了啊。", "教教我好不好？"]),
    ("T15", "厨房里，你第一次尝试自己做饭，锅里冒出了焦味。", "同住的监护人",
     ["哎呀，好像烧焦了。", "第一次做饭吗？", "要不要我帮忙？", "其实焦一点也能吃。", "下次我教你吧。"]),
    ("T16", "海边的修学旅行，同学们都在玩水，你一个人站在岸边。", "同班的女同学",
     ["一起下水玩吧！", "你不会游泳吗？", "站在这里不热吗？", "我给你拍张照吧。", "难得出来玩，开心一点嘛。"]),
    ("T17", "你被告知训练计划临时取消，可以自由活动一天。", "作战指挥官",
     ["今天放你一天假。", "想去哪里都可以。", "别总是待在宿舍里。", "要不要和大家一起去吃饭？", "明天早上照常集合。"]),
    ("T18", "雨夜的便利店门口，有人和你一起躲雨。", "同队的少年驾驶员",
     ["你也被雨困住了？", "我只买到了一把伞。", "要不一起撑吧？", "你冷不冷？", "这把伞给你，我跑回去就好。"]),
    ("T19", "一场模拟战你输给了新来的队员。", "新来的驾驶员",
     ["承让了。", "你刚才最后那一下有点慢。", "要不要再来一局？", "听说你是这里最强的？", "下次请多指教。"]),
    ("T20", "你在图书馆看书时，有人坐到了对面。", "同队的少年驾驶员",
     ["你在看什么书？", "这里可以坐吗？", "你经常来图书馆吗？", "这本书好看吗？", "我也想找本书看看。"]),
    ("T21", "家长参观日，同学们的家人都来了，只有你没有人来。", "同班同学",
     ["你家里人没来吗？", "要不要和我家人一起吃午饭？", "我妈做的便当很好吃哦。", "你不难过吗？", "下次参观日他们会来吗？"]),
    ("T22", "一次失误之后，你在走廊里被叫住。", "作战指挥官",
     ["刚才的失误差点害了整个小队。", "你知道自己错在哪里吗？", "写一份报告给我。", "这不像你平时的水平。", "我不是在怪你。"]),
    ("T23", "清晨的跑道上，你在晨练，有人跟上来和你并排跑。", "同队的少年驾驶员",
     ["早啊，你每天都这么早吗？", "我可以跟你一起跑吗？", "你跑得好快。", "跑完要不要去吃早饭？", "我快跑不动了。"]),
    ("T24", "你收到一封写着你名字的信，有人站在旁边看着你拆。", "同班的女同学",
     ["这是情书吧？", "是谁写的？", "你打算怎么回复？", "快读给我听听！", "你好像一点都不惊讶。"]),
]


def build():
    rng = random.Random(SEED)
    ids = [t[0] for t in TEMPLATES]
    test_templates = set(rng.sample(ids, TEST_TEMPLATES))
    train, test = [], []
    for template, situation, speaker, lines in TEMPLATES:
        for index, line in enumerate(lines, 1):
            for character in CHARACTERS:
                row = dict(id=f"{template}-L{index}-{character}", template=template, character=character,
                           situation=situation, speaker=speaker, line=line)
                (test if template in test_templates else train).append(row)
    return train, test, sorted(test_templates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({t[0] for t in TEMPLATES}) != len(TEMPLATES) or any(len(t[3]) != 5 for t in TEMPLATES):
        raise ValueError("template ids must be unique and each template needs five lines")
    train, test, test_templates = build()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "train.json", train)
    dump(args.output / "test.json", test)
    dump(args.output / "freeze.json", dict(
        files={name: digest(args.output / name) for name in ("train.json", "test.json")},
        persona_sha256=persona_identity(), split="by_template", seed=SEED,
        test_templates=test_templates, rows=dict(train=len(train), test=len(test)),
        source="original situations written for this project; no dialogue from the original work"))
    load_suite(args.output)
    print(json.dumps(dict(train=len(train), test=len(test), test_templates=test_templates), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
