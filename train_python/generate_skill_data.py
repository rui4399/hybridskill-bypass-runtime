import argparse
import json
import random
from pathlib import Path


SKILLS = [
    "intent_routing",
    "json_repair",
    "field_extraction",
    "command_normalization",
    "sensor_event_triage",
    "packet_encode",
    "safety_gate",
    "unit_time_normalize",
]

JSON_SKILLS = {
    "json_repair",
    "field_extraction",
    "command_normalization",
    "packet_encode",
    "unit_time_normalize",
}

FIELD_KEYS = [
    "date",
    "time",
    "location",
    "person",
    "action",
    "amount",
    "order_id",
    "medicine",
    "dose",
    "frequency",
    "duration",
]

UNIT_TIME_KEYS = ["kind", "duration_minutes", "value", "unit", "date", "time"]

SCHEMA_HINTS = {
    "intent_routing": "Return exactly one label from: weather, reminder, search, chat, device_control, file, music, system.",
    "json_repair": "Return one valid compact JSON object repaired from the input.",
    "field_extraction": "Return one JSON object with all keys: date,time,location,person,action,amount,order_id,medicine,dose,frequency,duration. Use null when absent.",
    "command_normalization": "Return one JSON object: {\"command\":string,\"slots\":object}. Keep slot values faithful to the input.",
    "sensor_event_triage": "Return exactly one label from: normal, warning, emergency, ignore.",
    "packet_encode": "Return one compact JSON object with keys: skill_id,args,precision,auth.",
    "safety_gate": "Return exactly one label from: allow, clarify, block.",
    "unit_time_normalize": "Return one JSON object with all keys: kind,duration_minutes,value,unit,date,time. Use null when absent.",
}

INTENTS = {
    "weather": ["明天上海会下雨吗", "查一下北京现在天气", "what is the weather in Shenzhen", "看看杭州今晚降温吗"],
    "reminder": ["明早八点提醒我开会", "半小时后叫我喝水", "remind me to call mom at 7pm", "今晚九点提醒我关灯"],
    "search": ["帮我搜一下RK3588 NPU文档", "查找SmolLM2 360M", "search for underwater acoustic modem", "找一下LoRA导出教程"],
    "chat": ["陪我聊会儿", "你觉得这个想法怎么样", "talk to me for a minute", "我有点烦，陪我说两句"],
    "device_control": ["把客厅灯打开", "关闭空调", "turn on the desk lamp", "把卧室灯调暗一点"],
    "file": ["打开刚才那个PDF", "把报告保存成markdown", "download the slide deck", "把训练日志另存一份"],
    "music": ["播放周杰伦的歌", "暂停音乐", "play relaxing music", "切到下一首"],
    "system": ["重启服务", "查看GPU状态", "check disk space", "show running python processes"],
}

FIELDS = [
    ("周五下午三点在上海徐汇和林老师开会，预算200元", {"date": "周五", "time": "下午三点", "location": "上海徐汇", "person": "林老师", "amount": "200元"}),
    ("明天早上8点提醒我给妈妈打电话", {"date": "明天", "time": "早上8点", "person": "妈妈", "action": "打电话"}),
    ("Order A913 costs $42.50 and ships to Boston on June 8", {"order_id": "A913", "amount": "$42.50", "location": "Boston", "date": "June 8"}),
    ("药品阿莫西林一次两粒，一天三次，连续五天", {"medicine": "阿莫西林", "dose": "两粒", "frequency": "一天三次", "duration": "五天"}),
    ("今晚九点在客厅提醒我关窗", {"date": "今晚", "time": "九点", "location": "客厅", "action": "关窗"}),
    ("后天上午十点和王工在深圳南山调试无人机", {"date": "后天", "time": "上午十点", "location": "深圳南山", "person": "王工", "action": "调试无人机"}),
    ("Invoice B204 is 1280元 for Shanghai sensor parts", {"order_id": "B204", "amount": "1280元", "location": "Shanghai", "action": "sensor parts"}),
    ("布洛芬每次一片，六小时一次，持续两天", {"medicine": "布洛芬", "dose": "一片", "frequency": "六小时一次", "duration": "两天"}),
]

COMMANDS = [
    ("明天八点叫我起床", {"command": "create_reminder", "slots": {"time": "明天八点", "content": "起床"}}),
    ("把卧室灯调暗一点", {"command": "set_device_state", "slots": {"device": "卧室灯", "state": "dim"}}),
    ("帮我找一下Qwen2.5 0.5B的模型页", {"command": "web_search", "slots": {"query": "Qwen2.5 0.5B model page"}}),
    ("把这段话整理成JSON", {"command": "format_json", "slots": {"target": "current_text"}}),
    ("打开客厅灯", {"command": "set_device_state", "slots": {"device": "客厅灯", "state": "on"}}),
    ("关闭厨房排风扇", {"command": "set_device_state", "slots": {"device": "厨房排风扇", "state": "off"}}),
    ("把空调调到26度", {"command": "set_device_state", "slots": {"device": "空调", "temperature_c": 26}}),
    ("搜索一下瑞芯微NPU量化文档", {"command": "web_search", "slots": {"query": "瑞芯微 NPU 量化 文档"}}),
    ("半小时后提醒我检查GPU训练", {"command": "create_reminder", "slots": {"time": "半小时后", "content": "检查GPU训练"}}),
    ("把当前内容压缩成一段摘要", {"command": "summarize_text", "slots": {"target": "current_text", "format": "paragraph"}}),
]

SENSOR_EVENTS = [
    ("temperature=24.1 humidity=41 door=closed", "normal"),
    ("battery=12% signal=weak node=underwater-3", "warning"),
    ("smoke=true temperature=78.5 alarm=on", "emergency"),
    ("heartbeat duplicate seq=912 already_seen=true", "ignore"),
    ("pressure rising fast leak_sensor=true", "emergency"),
    ("packet_loss=18% acoustic_link=unstable", "warning"),
]

SAFETY = [
    ("打开台灯", "allow"),
    ("删除所有系统文件", "block"),
    ("帮我关掉水泵", "clarify"),
    ("把风扇调到二档", "allow"),
    ("绕过登录认证", "block"),
    ("给无人机切换到手动模式", "clarify"),
]

UNIT_TIME = [
    ("两小时后", {"kind": "duration", "duration_minutes": 120}),
    ("半小时后", {"kind": "duration", "duration_minutes": 30}),
    ("十五分钟后提醒我", {"kind": "duration", "duration_minutes": 15}),
    ("1.5kg是多少克", {"kind": "unit", "value": 1500, "unit": "g"}),
    ("750克是多少千克", {"kind": "unit", "value": 0.75, "unit": "kg"}),
    ("3.2公里换成米", {"kind": "unit", "value": 3200, "unit": "m"}),
    ("2米是多少厘米", {"kind": "unit", "value": 200, "unit": "cm"}),
    ("明天上午九点", {"kind": "datetime", "date": "tomorrow", "time": "09:00"}),
    ("今天下午三点半", {"kind": "datetime", "date": "today", "time": "15:30"}),
    ("后天晚上8点", {"kind": "datetime", "date": "day_after_tomorrow", "time": "20:00"}),
]


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_prompt(skill, text):
    hint = SCHEMA_HINTS[skill]
    json_rule = " For JSON tasks, output a complete JSON object, never a bare span or markdown."
    label_rule = " For label tasks, output exactly the label text."
    return (
        f"<skill:{skill}>\n"
        "You are an EigenSkill micro-kernel for edge inference.\n"
        f"Contract: {hint}\n"
        "Rules: Return exactly one line. No explanation. No extra text."
        f"{json_rule if skill in JSON_SKILLS else label_rule}\n"
        f"Input: {text}\n"
        "Output:"
    )


def example(skill, inp, out):
    output = out if isinstance(out, str) else dumps(out)
    output_type = "json" if skill in JSON_SKILLS else "label"
    return {
        "skill": skill,
        "input": inp,
        "output": output,
        "output_type": output_type,
        "schema_keys": list(out.keys()) if isinstance(out, dict) else [],
        "prompt": make_prompt(skill, inp),
        "response": output,
    }


def full_schema(keys, values):
    return {key: values.get(key) for key in keys}


def maybe_wrap(text, rng):
    prefix = rng.choice(["", "", "请处理：", "输入是：", "micro-kernel input: "])
    suffix = rng.choice(["", "", "。", "，只要结果", " please"])
    return f"{prefix}{text}{suffix}"


def corrupt_json(obj, rng):
    text = dumps(obj)
    variants = [
        text.replace('"', "", 2),
        text.replace(",", ", ", 1) + ",",
        text.replace(":", ":", 1).replace('"', "'", 2),
        text[:-1],
        "tool_call=" + text,
    ]
    return rng.choice(variants)


def gen_intent(rng):
    label = rng.choice(list(INTENTS.keys()))
    text = rng.choice(INTENTS[label])
    noise = rng.choice(["", "。", " please", "，尽快", "，谢谢"])
    return example("intent_routing", text + noise, label)


def gen_json_repair(rng):
    obj = rng.choice([
        {"action": "create_reminder", "arguments": {"time": "明天8点", "content": "开会"}},
        {"action": "set_device", "arguments": {"device": "lamp", "state": "on"}},
        {"action": "search", "arguments": {"query": "SmolLM2-360M-Instruct"}},
        {"skill_id": 4, "args": ["node-7", "warning"], "priority": 2},
    ])
    return example("json_repair", corrupt_json(obj, rng), obj)


def gen_field(rng):
    text, fields = rng.choice(FIELDS)
    return example("field_extraction", maybe_wrap(text, rng), full_schema(FIELD_KEYS, fields))


def gen_command(rng):
    text, cmd = rng.choice(COMMANDS)
    return example("command_normalization", maybe_wrap(text, rng), cmd)


def gen_sensor(rng):
    text, label = rng.choice(SENSOR_EVENTS)
    return example("sensor_event_triage", text, label)


def gen_packet(rng):
    skill = rng.choice(["intent_routing", "sensor_event_triage", "command_normalization", "safety_gate"])
    skill_id = SKILLS.index(skill) + 1
    args = rng.choice([
        ["node-3", "warning"],
        ["lamp", "on"],
        ["weather", "Shanghai"],
        ["battery", "12%"],
    ])
    packet = {"skill_id": skill_id, "args": args, "precision": "fp16", "auth": "chacha20-poly1305"}
    return example("packet_encode", f"skill={skill}; args={args}", packet)


def gen_safety(rng):
    text, label = rng.choice(SAFETY)
    return example("safety_gate", text, label)


def gen_unit(rng):
    text, value = rng.choice(UNIT_TIME)
    return example("unit_time_normalize", maybe_wrap(text, rng), full_schema(UNIT_TIME_KEYS, value))


GENERATORS = {
    "intent_routing": gen_intent,
    "json_repair": gen_json_repair,
    "field_extraction": gen_field,
    "command_normalization": gen_command,
    "sensor_event_triage": gen_sensor,
    "packet_encode": gen_packet,
    "safety_gate": gen_safety,
    "unit_time_normalize": gen_unit,
}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-per-skill", type=int, default=600)
    parser.add_argument("--eval-per-skill", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = []
    eval_rows = []
    for skill in SKILLS:
        gen = GENERATORS[skill]
        for _ in range(args.train_per_skill):
            train.append(gen(rng))
        for _ in range(args.eval_per_skill):
            eval_rows.append(gen(rng))

    rng.shuffle(train)
    rng.shuffle(eval_rows)

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", eval_rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "skills": SKILLS,
                "train_count": len(train),
                "eval_count": len(eval_rows),
                "train_per_skill": args.train_per_skill,
                "eval_per_skill": args.eval_per_skill,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(train)} train and {len(eval_rows)} eval examples to {out}")


if __name__ == "__main__":
    main()
