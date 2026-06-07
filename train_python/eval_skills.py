import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    return text.strip().splitlines()[0].strip()


def first_json_value(text):
    text = text.strip()
    start = -1
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start < 0:
        return None

    stack = []
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            expected = "}" if stack[-1] == "{" else "]"
            if char != expected:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def parse_json_output(text):
    candidate = first_json_value(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_json_gold(text):
    text = text.strip()
    return text.startswith("{") or text.startswith("[")


def schema_ok(value, keys):
    if not keys:
        return True
    if not isinstance(value, dict):
        return False
    return all(key in value for key in keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    stats = defaultdict(lambda: {"n": 0, "exact": 0, "json_valid": 0, "schema_ok": 0})
    predictions = []

    for row in rows:
        prompt = row["prompt"] + " "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        gold_raw = row["response"].strip()
        pred_value = None
        if is_json_gold(gold_raw):
            gold_value = json.loads(gold_raw)
            pred_value = parse_json_output(generated)
            pred = canonical_json(pred_value) if pred_value is not None else normalize(generated)
            gold = canonical_json(gold_value)
        else:
            pred = normalize(generated)
            gold = normalize(gold_raw)
        skill = row["skill"]
        stats[skill]["n"] += 1
        stats[skill]["exact"] += int(pred == gold)
        if is_json_gold(gold_raw):
            stats[skill]["json_valid"] += int(pred_value is not None)
            stats[skill]["schema_ok"] += int(schema_ok(pred_value, row.get("schema_keys", [])))
        predictions.append(
            {
                "skill": skill,
                "input": row["input"],
                "gold": gold,
                "pred": pred,
                "raw_pred": generated.strip(),
                "json_valid": pred_value is not None if is_json_gold(gold_raw) else None,
                "schema_ok": schema_ok(pred_value, row.get("schema_keys", [])) if is_json_gold(gold_raw) else None,
            }
        )

    summary = {}
    for skill, value in stats.items():
        n = max(value["n"], 1)
        summary[skill] = {
            "n": value["n"],
            "exact": value["exact"] / n,
            "json_valid": value["json_valid"] / n,
            "schema_ok": value["schema_ok"] / n,
        }

    result = {"summary": summary, "predictions": predictions}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
