#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DECISION_KEYS = {
    "outlier_detect": ["protect", "policy"],
    "bit_allocate": ["bits", "format"],
    "rotation_select": ["rotation", "format"],
    "residual_patch": ["rank", "apply"],
    "kv_policy": ["policy", "context"],
}


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def schema_ok(pred, gold):
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    return all(key in pred for key in gold.keys())


def decision_ok(skill, pred, gold):
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    keys = DECISION_KEYS.get(skill, [])
    return all(pred.get(key) == gold.get(key) for key in keys)


def update_key_stats(stats, skill, pred, gold):
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        for key in gold.keys():
            stats[skill]["field_total"][key] += 1
        return
    for key, value in gold.items():
        stats[skill]["field_total"][key] += 1
        stats[skill]["field_exact"][key] += int(pred.get(key) == value)


def summarize(stats):
    summary = {}
    global_counts = defaultdict(int)
    global_field_total = defaultdict(int)
    global_field_exact = defaultdict(int)

    for skill, value in sorted(stats.items()):
        n = max(value["n"], 1)
        field_accuracy = {}
        for key, total in sorted(value["field_total"].items()):
            field_accuracy[key] = value["field_exact"][key] / max(total, 1)
            global_field_total[key] += total
            global_field_exact[key] += value["field_exact"][key]
        for key in ["n", "json_valid", "schema_ok", "exact_json", "decision_exact"]:
            global_counts[key] += value[key]
        summary[skill] = {
            "n": value["n"],
            "json_valid": value["json_valid"] / n,
            "schema_ok": value["schema_ok"] / n,
            "exact_json": value["exact_json"] / n,
            "decision_exact": value["decision_exact"] / n,
            "field_accuracy": field_accuracy,
        }

    total_n = max(global_counts["n"], 1)
    summary["_overall"] = {
        "n": global_counts["n"],
        "json_valid": global_counts["json_valid"] / total_n,
        "schema_ok": global_counts["schema_ok"] / total_n,
        "exact_json": global_counts["exact_json"] / total_n,
        "decision_exact": global_counts["decision_exact"] / total_n,
        "field_accuracy": {
            key: global_field_exact[key] / max(total, 1)
            for key, total in sorted(global_field_total.items())
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
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

    stats = defaultdict(
        lambda: {
            "n": 0,
            "json_valid": 0,
            "schema_ok": 0,
            "exact_json": 0,
            "decision_exact": 0,
            "field_total": defaultdict(int),
            "field_exact": defaultdict(int),
        }
    )
    predictions = []
    errors = []

    for index, row in enumerate(rows, start=1):
        prompt = row["prompt"] + " "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        skill = row["skill"]
        gold = json.loads(row["response"])
        pred = parse_json_output(generated)
        exact = pred is not None and canonical_json(pred) == canonical_json(gold)
        valid = pred is not None
        schema = schema_ok(pred, gold)
        decision = decision_ok(skill, pred, gold)

        stats[skill]["n"] += 1
        stats[skill]["json_valid"] += int(valid)
        stats[skill]["schema_ok"] += int(schema)
        stats[skill]["exact_json"] += int(exact)
        stats[skill]["decision_exact"] += int(decision)
        update_key_stats(stats, skill, pred, gold)

        record = {
            "skill": skill,
            "input": row["input"],
            "gold": gold,
            "pred": pred,
            "raw_pred": generated,
            "json_valid": valid,
            "schema_ok": schema,
            "exact_json": exact,
            "decision_exact": decision,
        }
        predictions.append(record)
        if (not decision or not schema or not exact) and len(errors) < 40:
            errors.append(record)
        if args.print_every and index % args.print_every == 0:
            print(json.dumps({"processed": index, "total": len(rows)}, ensure_ascii=False), flush=True)

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "limit": args.limit,
        "summary": summarize(stats),
        "errors": errors,
        "predictions": predictions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
