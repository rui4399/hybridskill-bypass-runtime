#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from generate_quant_skill_data import (
    choose_bits,
    choose_kv,
    choose_outlier,
    choose_residual,
    choose_rotation,
)


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


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def strip_layer(text):
    return re.sub(r"layer_\d+", "layer", text)


def strip_format(text):
    return re.sub(r"\b(?:INT4|INT3|MXFP4|NVFP4)\b", "FORMAT", text, flags=re.IGNORECASE)


def floats_without_layer(text):
    text = strip_format(strip_layer(text))
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def first_match(patterns, text, cast=float):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return cast(match.group(1))
    return None


def parse_budget(text):
    match = re.search(r"\b(tight|medium|relaxed)\b", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"budget not found: {text}")
    return match.group(1).lower()


def parse_format(text):
    match = re.search(r"\b(INT4|INT3|MXFP4|NVFP4)\b", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"format not found: {text}")
    return match.group(1).upper()


def predict(skill, text):
    nums = floats_without_layer(text)
    if skill == "outlier_detect":
        mx = first_match(
            [
                r"\bmax\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)",
                r"\bpeak\s+([-+]?\d+(?:\.\d+)?)",
                r"\bchannel_peak\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
            ],
            text,
        )
        p99 = first_match(
            [
                r"\bp99\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)",
                r"\btail_p99\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
            ],
            text,
        )
        kurt = first_match(
            [
                r"\bkurtosis\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)",
                r"\bkurt\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
            ],
            text,
        )
        if mx is None or p99 is None or kurt is None:
            raise ValueError(f"outlier numbers not found: {text}")
        return choose_outlier(mx, p99, kurt)
    if skill == "bit_allocate":
        if len(nums) < 2:
            raise ValueError(f"bit allocation numbers not found: {text}")
        return choose_bits(nums[0], nums[1], parse_budget(text))
    if skill == "rotation_select":
        if len(nums) < 2:
            raise ValueError(f"rotation numbers not found: {text}")
        return choose_rotation(nums[0], nums[1], parse_format(text))
    if skill == "residual_patch":
        if len(nums) < 3:
            raise ValueError(f"residual spectrum not found: {text}")
        return choose_residual(nums[0], nums[1], nums[2])
    if skill == "kv_policy":
        ctx = first_match(
            [
                r"\bctx\s*[=:]\s*(\d+)",
                r"\bcontext\s+(\d+)",
                r"\blength\s+(\d+)",
                r"\bctx\s+(\d+)",
            ],
            text,
            int,
        )
        ks = first_match(
            [
                r"\bK\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
                r"\bK\s+sensitivity\s+([-+]?\d+(?:\.\d+)?)",
                r"\bkey_sens\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
                r"\bkey\s+sens\s+([-+]?\d+(?:\.\d+)?)",
            ],
            text,
        )
        vs = first_match(
            [
                r"\bV\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
                r"\bV\s+sensitivity\s+([-+]?\d+(?:\.\d+)?)",
                r"\bvalue_sens\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
                r"\bvalue\s+sens\s+([-+]?\d+(?:\.\d+)?)",
            ],
            text,
        )
        if ctx is None or ks is None or vs is None:
            raise ValueError(f"kv parameters not found: {text}")
        return choose_kv(ctx, ks, vs)
    raise ValueError(f"unknown skill: {skill}")


def decision_ok(skill, pred, gold):
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    return all(pred.get(key) == gold.get(key) for key in DECISION_KEYS.get(skill, []))


def update_field_stats(bucket, pred, gold):
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        for key in gold.keys():
            bucket["field_total"][key] += 1
        return
    for key, value in gold.items():
        bucket["field_total"][key] += 1
        bucket["field_exact"][key] += int(pred.get(key) == value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    stats = defaultdict(
        lambda: {
            "n": 0,
            "exact_json": 0,
            "decision_exact": 0,
            "parse_error": 0,
            "field_total": defaultdict(int),
            "field_exact": defaultdict(int),
        }
    )
    predictions = []
    errors = []

    for row in rows:
        skill = row["skill"]
        gold = json.loads(row["response"])
        stats[skill]["n"] += 1
        try:
            pred = predict(skill, row["input"])
            parse_error = False
        except Exception as exc:
            pred = {"error": str(exc)}
            parse_error = True
        exact_json = (not parse_error) and canonical_json(pred) == canonical_json(gold)
        decision_exact = (not parse_error) and decision_ok(skill, pred, gold)
        stats[skill]["exact_json"] += int(exact_json)
        stats[skill]["decision_exact"] += int(decision_exact)
        stats[skill]["parse_error"] += int(parse_error)
        update_field_stats(stats[skill], pred, gold)
        record = {
            "skill": skill,
            "input": row["input"],
            "gold": gold,
            "pred": pred,
            "path": "deterministic_quant_bypass",
            "exact_json": exact_json,
            "decision_exact": decision_exact,
            "parse_error": parse_error,
        }
        predictions.append(record)
        if (not decision_exact or not exact_json) and len(errors) < 40:
            errors.append(record)

    summary = {}
    total_n = 0
    total_exact_json = 0
    total_decision_exact = 0
    total_parse_error = 0
    total_field_total = defaultdict(int)
    total_field_exact = defaultdict(int)
    for skill, value in sorted(stats.items()):
        n = max(value["n"], 1)
        field_accuracy = {}
        for key, total in sorted(value["field_total"].items()):
            field_accuracy[key] = value["field_exact"][key] / max(total, 1)
            total_field_total[key] += total
            total_field_exact[key] += value["field_exact"][key]
        summary[skill] = {
            "n": value["n"],
            "exact_json": value["exact_json"] / n,
            "decision_exact": value["decision_exact"] / n,
            "parse_error": value["parse_error"] / n,
            "field_accuracy": field_accuracy,
        }
        total_n += value["n"]
        total_exact_json += value["exact_json"]
        total_decision_exact += value["decision_exact"]
        total_parse_error += value["parse_error"]
    summary["_overall"] = {
        "n": total_n,
        "exact_json": total_exact_json / max(total_n, 1),
        "decision_exact": total_decision_exact / max(total_n, 1),
        "parse_error": total_parse_error / max(total_n, 1),
        "field_accuracy": {
            key: total_field_exact[key] / max(total, 1)
            for key, total in sorted(total_field_total.items())
        },
    }

    result = {
        "data": args.data,
        "limit": args.limit,
        "summary": summary,
        "errors": errors,
        "predictions": predictions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
