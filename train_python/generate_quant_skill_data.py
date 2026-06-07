#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SKILLS = ["outlier_detect", "bit_allocate", "rotation_select", "residual_patch", "kv_policy"]

TRAIN_TEMPLATES = {
    "outlier_detect": [
        "Given layer {layer} stats max={mx:.2f}, p99={p99:.2f}, kurtosis={kurt:.2f}, mark risky quantization channels.",
        "Quant audit for {layer}: channel_peak={mx:.2f}, tail_p99={p99:.2f}, kurt={kurt:.2f}. Return outlier policy JSON.",
    ],
    "bit_allocate": [
        "Allocate bits for {layer}: sensitivity={sens:.3f}, variance={var:.3f}, budget={budget}. Return JSON.",
        "For layer {layer}, S={sens:.3f}, sigma2={var:.3f}, memory budget class={budget}; choose bit width.",
    ],
    "rotation_select": [
        "Choose rotation for {layer}: block_var={block:.3f}, kurtosis={kurt:.2f}, format={fmt}.",
        "Layer {layer} has block imbalance {block:.3f} and tail {kurt:.2f} under {fmt}. Return rotation policy.",
    ],
    "residual_patch": [
        "Residual spectrum for {layer}: e1={e1:.3f}, e4={e4:.3f}, e8={e8:.3f}. Select low-rank patch.",
        "Quant error energy on {layer}: top1={e1:.3f}, top4={e4:.3f}, top8={e8:.3f}; return residual rank JSON.",
    ],
    "kv_policy": [
        "KV cache stats: ctx={ctx}, key_sens={ks:.3f}, value_sens={vs:.3f}, head={head}. Choose KV quant policy.",
        "For attention head {head}, context {ctx}, K sensitivity {ks:.3f}, V sensitivity {vs:.3f}; return cache policy.",
    ],
}

EVAL_TEMPLATES = {
    "outlier_detect": [
        "Assess quantization risk in {layer} with max {mx:.2f}, p99 {p99:.2f}, and kurtosis {kurt:.2f}. JSON only.",
    ],
    "bit_allocate": [
        "Under budget {budget}, decide precision for {layer} where sensitivity is {sens:.3f} and variance is {var:.3f}.",
    ],
    "rotation_select": [
        "For {fmt} quantization, {layer} shows variance imbalance {block:.3f} and kurtosis {kurt:.2f}; choose rotation.",
    ],
    "residual_patch": [
        "Select compensation rank for {layer} from spectrum energies {e1:.3f}/{e4:.3f}/{e8:.3f}. JSON only.",
    ],
    "kv_policy": [
        "Choose K/V cache compression for head {head}: length {ctx}, K={ks:.3f}, V={vs:.3f}. JSON only.",
    ],
}

TEST_TEMPLATES = {
    "outlier_detect": [
        "Layer {layer} quant stats are peak {mx:.2f}, p99 {p99:.2f}, kurtosis {kurt:.2f}. Which channels need protection?",
    ],
    "bit_allocate": [
        "Precision routing request: {layer}, sens {sens:.3f}, var {var:.3f}, budget tier {budget}. Produce policy JSON.",
    ],
    "rotation_select": [
        "Rotation routing request for {layer}: format {fmt}, block variance {block:.3f}, tail kurtosis {kurt:.2f}.",
    ],
    "residual_patch": [
        "Low-rank compensation request on {layer}: spectrum top energies {e1:.3f}, {e4:.3f}, {e8:.3f}.",
    ],
    "kv_policy": [
        "Cache routing request: attention head {head}, ctx {ctx}, key sens {ks:.3f}, value sens {vs:.3f}.",
    ],
}

LAYERS = [f"layer_{i}" for i in range(4, 28)]
FORMATS = ["INT4", "INT3", "MXFP4", "NVFP4"]
BUDGETS = ["tight", "medium", "relaxed"]


def choose_outlier(mx, p99, kurt):
    risk = (mx / max(p99, 1e-6)) + 0.25 * kurt
    protect = risk > 5.2 or kurt > 9.0 or mx > 8.0
    return {"skill": "outlier_detect", "protect": protect, "risk": round(risk, 3), "policy": "protect_top_channels" if protect else "standard_group_quant"}


def choose_bits(sens, var, budget):
    score = sens * math.sqrt(var)
    if budget == "tight":
        bits = 4 if score > 1.45 else 3 if score > 0.8 else 2
    elif budget == "medium":
        bits = 8 if score > 2.2 else 4 if score > 0.75 else 3
    else:
        bits = 8 if score > 1.25 else 4
    return {"skill": "bit_allocate", "bits": bits, "score": round(score, 3), "format": "INT" + str(bits) if bits != 8 else "INT8"}


def choose_rotation(block, kurt, fmt):
    if fmt in {"MXFP4", "NVFP4"} and block > 1.35:
        rot = "two_level_block_orthogonal"
    elif kurt > 7.5:
        rot = "block_hadamard"
    elif block > 1.1:
        rot = "butterfly"
    else:
        rot = "none"
    return {"skill": "rotation_select", "rotation": rot, "format": fmt}


def choose_residual(e1, e4, e8):
    if e8 > 0.78:
        rank = 8
    elif e4 > 0.62:
        rank = 4
    elif e1 > 0.45:
        rank = 2
    else:
        rank = 0
    return {"skill": "residual_patch", "rank": rank, "apply": rank > 0}


def choose_kv(ctx, ks, vs):
    sens = max(ks, vs)
    if ctx >= 8192 and sens < 0.55:
        policy = "int2_history_bf16_recent"
    elif ctx >= 4096 and sens < 0.8:
        policy = "int4_tokenwise_with_recent_window"
    elif sens >= 1.1:
        policy = "int8_or_bf16_sensitive_heads"
    else:
        policy = "int4_groupwise"
    return {"skill": "kv_policy", "policy": policy, "context": ctx}


def sample_params(skill, rng, split):
    # Deliberately shift numeric ranges by split to reduce train/eval/test overlap.
    offset = {"train": 0.0, "eval": 0.37, "test": 0.73}[split]
    layer = rng.choice(LAYERS)
    if skill == "outlier_detect":
        return {"layer": layer, "mx": rng.uniform(2.0+offset, 11.0+offset), "p99": rng.uniform(0.8, 2.4), "kurt": rng.uniform(2.0+offset, 14.0+offset)}
    if skill == "bit_allocate":
        return {"layer": layer, "sens": rng.uniform(0.05+offset/10, 2.8+offset/10), "var": rng.uniform(0.08, 3.2), "budget": rng.choice(BUDGETS)}
    if skill == "rotation_select":
        return {"layer": layer, "block": rng.uniform(0.55+offset/10, 1.9+offset/10), "kurt": rng.uniform(2.0, 13.0+offset), "fmt": rng.choice(FORMATS)}
    if skill == "residual_patch":
        e1 = rng.uniform(0.05, 0.65)
        e4 = min(0.98, e1 + rng.uniform(0.05, 0.35))
        e8 = min(0.99, e4 + rng.uniform(0.04, 0.25))
        return {"layer": layer, "e1": e1, "e4": e4, "e8": e8}
    if skill == "kv_policy":
        return {"ctx": rng.choice([1024, 2048, 4096, 8192, 16384]), "ks": rng.uniform(0.15+offset/10, 1.4+offset/10), "vs": rng.uniform(0.15, 1.4), "head": rng.randint(0, 31)}
    raise ValueError(skill)


def round_visible(skill, p):
    p = dict(p)
    if skill == "outlier_detect":
        p["mx"] = round(p["mx"], 2)
        p["p99"] = round(p["p99"], 2)
        p["kurt"] = round(p["kurt"], 2)
    elif skill == "bit_allocate":
        p["sens"] = round(p["sens"], 3)
        p["var"] = round(p["var"], 3)
    elif skill == "rotation_select":
        p["block"] = round(p["block"], 3)
        p["kurt"] = round(p["kurt"], 2)
    elif skill == "residual_patch":
        p["e1"] = round(p["e1"], 3)
        p["e4"] = round(p["e4"], 3)
        p["e8"] = round(p["e8"], 3)
    elif skill == "kv_policy":
        p["ks"] = round(p["ks"], 3)
        p["vs"] = round(p["vs"], 3)
    else:
        raise ValueError(skill)
    return p


def label(skill, p):
    if skill == "outlier_detect": return choose_outlier(p["mx"], p["p99"], p["kurt"])
    if skill == "bit_allocate": return choose_bits(p["sens"], p["var"], p["budget"])
    if skill == "rotation_select": return choose_rotation(p["block"], p["kurt"], p["fmt"])
    if skill == "residual_patch": return choose_residual(p["e1"], p["e4"], p["e8"])
    if skill == "kv_policy": return choose_kv(p["ctx"], p["ks"], p["vs"])
    raise ValueError(skill)


def make_row(skill, rng, split):
    templates = {"train": TRAIN_TEMPLATES, "eval": EVAL_TEMPLATES, "test": TEST_TEMPLATES}[split]
    p = round_visible(skill, sample_params(skill, rng, split))
    text = rng.choice(templates[skill]).format(**p)
    response = json.dumps(label(skill, p), ensure_ascii=False, sort_keys=True)
    return {"skill": skill, "split": split, "input": text, "prompt": f"<quant_skill:{skill}> {text}\nReturn compact JSON.", "response": response, "template_family": split}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit(splits):
    result = {"counts": {k: len(v) for k, v in splits.items()}}
    for a, b in [("train", "eval"), ("train", "test"), ("eval", "test")]:
        exact_a = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in splits[a]}
        exact_b = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in splits[b]}
        inputs_a = {x["input"] for x in splits[a]}
        inputs_b = {x["input"] for x in splits[b]}
        result[f"{a}_{b}_exact_overlap"] = len(exact_a & exact_b)
        result[f"{a}_{b}_input_overlap"] = len(inputs_a & inputs_b)
    result["per_skill"] = {}
    for split, rows in splits.items():
        by = defaultdict(list)
        for r in rows: by[r["skill"]].append(r)
        result["per_skill"][split] = {k: len(v) for k, v in by.items()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_eval/eigenskill_quant_v0")
    ap.add_argument("--train-per-skill", type=int, default=180)
    ap.add_argument("--eval-per-skill", type=int, default=60)
    ap.add_argument("--test-per-skill", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1314)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    splits = {"train": [], "eval": [], "test": []}
    for skill in SKILLS:
        for _ in range(args.train_per_skill): splits["train"].append(make_row(skill, rng, "train"))
        for _ in range(args.eval_per_skill): splits["eval"].append(make_row(skill, rng, "eval"))
        for _ in range(args.test_per_skill): splits["test"].append(make_row(skill, rng, "test"))
    for rows in splits.values(): rng.shuffle(rows)
    for name, rows in splits.items(): write_jsonl(out / f"{name}.jsonl", rows)
    report = audit(splits)
    report.update({"skills": SKILLS, "seed": args.seed, "template_split": "disjoint_by_split"})
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
