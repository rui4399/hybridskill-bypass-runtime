#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def first_number(text: str) -> str:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return ""
    value = match.group(0)
    if value.endswith(".0"):
        value = value[:-2]
    return value


def first_choice(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else ""


def score_prediction(prediction: str, answer: str, answer_type: str) -> dict:
    if answer_type == "number":
        pred = first_number(prediction)
        gold = first_number(answer)
        return {"normalized_prediction": pred, "normalized_answer": gold, "exact": pred == gold}
    if answer_type == "choice":
        pred = first_choice(prediction)
        gold = first_choice(answer)
        return {"normalized_prediction": pred, "normalized_answer": gold, "exact": pred == gold}
    pred = normalize_text(prediction)
    gold = normalize_text(answer)
    return {"normalized_prediction": pred, "normalized_answer": gold, "exact": pred == gold}


def generate(model, tokenizer, prompt: str, args) -> str:
    batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tasks", default="data_eval/task_prompts/quant_task_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default="outputs/task_eval_smoke_summary.json")
    args = parser.parse_args()

    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise SystemExit("This script requires torch and transformers.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    model.to(args.device)

    rows = read_jsonl(Path(args.tasks), args.limit)
    results = []
    for row in rows:
        prediction = generate(model, tokenizer, row["prompt"], args)
        scored = score_prediction(prediction, str(row["answer"]), row.get("answer_type", "exact"))
        results.append(
            {
                "id": row.get("id", ""),
                "task": row.get("task", ""),
                "answer_type": row.get("answer_type", "exact"),
                "answer": str(row["answer"]),
                "prediction": prediction,
                **scored,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    total = len(results)
    exact = sum(1 for row in results if row["exact"])
    by_task: dict[str, dict] = {}
    for row in results:
        bucket = by_task.setdefault(row["task"], {"total": 0, "exact": 0})
        bucket["total"] += 1
        bucket["exact"] += int(row["exact"])
    for bucket in by_task.values():
        bucket["accuracy"] = bucket["exact"] / max(bucket["total"], 1)

    out = Path(args.out)
    output = {
        "out": str(out),
        "model": args.model,
        "tasks": args.tasks,
        "device": args.device,
        "dtype": args.dtype,
        "total": total,
        "exact": exact,
        "accuracy": exact / max(total, 1),
        "by_task": by_task,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ["out", "total", "exact", "accuracy"] if k in output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
