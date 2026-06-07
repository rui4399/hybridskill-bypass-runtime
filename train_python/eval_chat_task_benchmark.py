#!/usr/bin/env python3
"""Run a small chat-template task benchmark with deterministic scoring."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _choice_answer(answer: Any, choices: list[Any]) -> str:
    if isinstance(answer, int):
        if 0 <= answer < len(CHOICE_LABELS):
            return CHOICE_LABELS[answer]
        raise ValueError(f"MMLU answer index out of range: {answer}")
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in CHOICE_LABELS:
        return text.upper()
    if text.isdigit():
        return _choice_answer(int(text), choices)
    for idx, choice in enumerate(choices):
        if str(choice).strip() == text:
            return CHOICE_LABELS[idx]
    raise ValueError(f"cannot map MMLU answer to a choice label: {answer!r}")


def task_from_mmlu_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    choices = list(record.get("choices", []))
    if not choices:
        raise ValueError("MMLU record missing choices")
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("MMLU record missing question")
    options = "\n".join(f"{CHOICE_LABELS[i]}. {choice}" for i, choice in enumerate(choices))
    return {
        "id": record.get("id", f"mmlu_{idx}"),
        "type": "mcq",
        "prompt": f"Answer with only the option letter.\nQuestion: {question}\n{options}",
        "answer": _choice_answer(record.get("answer"), choices),
        "source_format": "mmlu",
        "subject": record.get("subject", ""),
    }


def _gsm8k_final_answer(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip().replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    if not numbers:
        raise ValueError("GSM8K answer has no numeric final answer")
    return numbers[-1].replace(",", "")


def task_from_gsm8k_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("GSM8K record missing question")
    return {
        "id": record.get("id", f"gsm8k_{idx}"),
        "type": "number",
        "prompt": f"Answer with only the final number.\n{question}",
        "answer": _gsm8k_final_answer(str(record.get("answer", ""))),
        "source_format": "gsm8k",
    }


def _ifeval_kwargs(record: dict[str, Any], instruction_index: int) -> dict[str, Any]:
    kwargs = record.get("kwargs", {})
    if isinstance(kwargs, list):
        if instruction_index < len(kwargs) and isinstance(kwargs[instruction_index], dict):
            return kwargs[instruction_index]
        return {}
    if isinstance(kwargs, dict):
        return kwargs
    return {}


def task_from_ifeval_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("IFEval record missing prompt")
    instruction_ids = list(record.get("instruction_id_list", []))
    checks: list[dict[str, Any]] = []
    for instruction_index, instruction_id in enumerate(instruction_ids):
        instruction_name = str(instruction_id)
        if instruction_name == "keywords:existence":
            kwargs = _ifeval_kwargs(record, instruction_index)
            keywords = kwargs.get("keywords", [])
            if not keywords:
                raise ValueError("IFEval keywords:existence record missing keywords")
            checks.append({"type": "contains_all", "answer": [str(item) for item in keywords]})
        elif instruction_name == "keywords:forbidden_words":
            kwargs = _ifeval_kwargs(record, instruction_index)
            words = kwargs.get("forbidden_words", kwargs.get("keywords", []))
            if not words:
                raise ValueError("IFEval keywords:forbidden_words record missing forbidden words")
            checks.append({"type": "contains_none", "answer": [str(item) for item in words]})
        elif instruction_name == "detectable_format:json_format":
            checks.append({"type": "json_valid", "answer": True})
        elif instruction_name == "length_constraints:number_sentences":
            kwargs = _ifeval_kwargs(record, instruction_index)
            count = kwargs.get("num_sentences", kwargs.get("number_sentences"))
            if count is None:
                raise ValueError("IFEval number_sentences record missing num_sentences")
            checks.append({"type": "sentence_count", "answer": {"count": int(count)}})
        elif instruction_name == "length_constraints:number_words":
            kwargs = _ifeval_kwargs(record, instruction_index)
            count = kwargs.get("num_words", kwargs.get("number_words"))
            if count is None:
                raise ValueError("IFEval number_words record missing num_words")
            checks.append({"type": "word_count", "answer": {"count": int(count)}})
    if len(checks) == 1:
        check = checks[0]
        return {
            "id": record.get("key", record.get("id", f"ifeval_{idx}")),
            "type": check["type"],
            "prompt": prompt,
            "answer": check["answer"],
            "source_format": "ifeval",
            "instruction_id_list": instruction_ids,
        }
    if checks:
        return {
            "id": record.get("key", record.get("id", f"ifeval_{idx}")),
            "type": "all_of",
            "prompt": prompt,
            "answer": checks,
            "source_format": "ifeval",
            "instruction_id_list": instruction_ids,
        }
    raise ValueError(f"unsupported IFEval instruction subset: {instruction_ids}")


def _convert_task(record: dict[str, Any], idx: int, task_format: str) -> dict[str, Any]:
    if task_format == "native":
        missing = [key for key in ("id", "type", "prompt", "answer") if key not in record]
        if missing:
            raise ValueError(f"missing required keys {missing}")
        return record
    if task_format == "mmlu":
        return task_from_mmlu_record(record, idx)
    if task_format == "gsm8k":
        return task_from_gsm8k_record(record, idx)
    if task_format == "ifeval":
        return task_from_ifeval_record(record, idx)
    raise ValueError(f"unsupported task format: {task_format}")


def load_tasks(path: Path, task_format: str = "native") -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        record = json.loads(line)
        try:
            tasks.append(_convert_task(record, line_no - 1, task_format))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def apply_task_limit(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("task limit must be >= 0")
    if limit == 0:
        return tasks
    return tasks[:limit]


def format_task_prompt(prompt: str, *, no_think: bool = False) -> str:
    rendered = str(prompt).strip()
    if no_think and "/no_think" not in rendered:
        rendered = f"{rendered}\n/no_think"
    return rendered


def _visible_text(text: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", " ", text or "", flags=re.IGNORECASE | re.DOTALL).strip()


def extract_choice_letter(text: str) -> str | None:
    raw = _visible_text(text)
    patterns = [
        r"\banswer\s*[:\-]?\s*\(?([A-D])\)?\b",
        r"\bcorrect\s+(?:option|answer)\s+(?:is\s+)?\(?([A-D])\)?\b",
        r"\b(?:choose|chose|select|selected|option)\s*\(?([A-D])\)?\b",
        r"^\s*\(?([A-D])\)?(?:[.)\s]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.I)
        if match:
            return match.group(1).upper()
    return None


def _normalize_number(value: str) -> Decimal | None:
    cleaned = str(value).replace(",", "").strip().rstrip(".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _numbers_in_text(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", _visible_text(text)):
        value = _normalize_number(match.group(0))
        if value is not None:
            values.append(value)
    return values


def _json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{.*?\}", text or "", flags=re.DOTALL):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _strict_json_value(text: str) -> Any | None:
    raw = _visible_text(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _sentences(text: str) -> list[str]:
    raw = _visible_text(text)
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r"[.!?]+(?:\s+|$)", raw) if part.strip()]
    if not parts and raw:
        return [raw]
    return parts


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", _visible_text(text), flags=re.UNICODE)


def score_task(task: dict[str, Any], generated_text: str) -> dict[str, Any]:
    task_type = str(task["type"]).lower()
    expected = task["answer"]
    visible_text = _visible_text(generated_text)
    if task_type == "mcq":
        predicted = extract_choice_letter(generated_text)
        wanted = str(expected).strip().upper()
        return {"passed": predicted == wanted, "predicted": predicted, "expected": wanted}
    if task_type == "number":
        wanted = _normalize_number(str(expected))
        predicted_numbers = _numbers_in_text(generated_text)
        return {
            "passed": wanted is not None and wanted in predicted_numbers,
            "predicted": [str(value) for value in predicted_numbers],
            "expected": str(wanted),
        }
    if task_type == "contains_all":
        keywords = [str(item).lower() for item in expected]
        lower = visible_text.lower()
        missing = [kw for kw in keywords if kw not in lower]
        return {"passed": not missing, "missing": missing, "expected": keywords}
    if task_type == "contains_none":
        forbidden = [str(item).lower() for item in expected]
        lower = visible_text.lower()
        present = [word for word in forbidden if word in lower]
        return {"passed": not present, "present": present, "expected_absent": forbidden}
    if task_type == "json_valid":
        value = _strict_json_value(generated_text)
        return {"passed": value is not None, "parsed_type": type(value).__name__ if value is not None else None}
    if task_type == "json_keys":
        required = {str(item).lower().replace("-", "_") for item in expected}
        for obj in _json_objects(generated_text):
            keys = {str(key).lower().replace("-", "_") for key in obj}
            if required.issubset(keys):
                return {"passed": True, "found": sorted(keys), "expected": sorted(required)}
        return {"passed": False, "found": [], "expected": sorted(required)}
    if task_type == "sentence_count":
        wanted = int(expected["count"] if isinstance(expected, dict) else expected)
        found = len(_sentences(generated_text))
        return {"passed": found == wanted, "found": found, "expected": wanted}
    if task_type == "word_count":
        wanted = int(expected["count"] if isinstance(expected, dict) else expected)
        found = len(_words(generated_text))
        return {"passed": found == wanted, "found": found, "expected": wanted}
    if task_type == "all_of":
        sub_scores = [score_task(subtask, generated_text) for subtask in expected]
        return {
            "passed": all(score["passed"] for score in sub_scores),
            "sub_scores": sub_scores,
            "expected": expected,
        }
    if task_type == "exact":
        cleaned = " ".join(visible_text.strip().split())
        wanted = " ".join(str(expected).strip().split())
        return {"passed": cleaned == wanted, "predicted": cleaned, "expected": wanted}
    raise ValueError(f"unsupported task type: {task_type}")


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = len(rows)
    passes = sum(1 for row in rows if row["score"]["passed"])
    token_rates = [float(row.get("tokens_per_second") or 0.0) for row in rows]
    ttfts = [float(row.get("ttft_seconds") or 0.0) for row in rows]
    return {
        "tasks": tasks,
        "passes": passes,
        "accuracy": passes / max(tasks, 1),
        "mean_tokens_per_second": mean(token_rates) if token_rates else 0.0,
        "mean_ttft_seconds": mean(ttfts) if ttfts else 0.0,
    }


def evaluate_rows(
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    max_new_tokens: int,
    use_chat_template: bool,
    no_think: bool = False,
) -> list[dict[str, Any]]:
    from measure_esmp_generation_latency import run_generation

    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = format_task_prompt(str(task["prompt"]), no_think=no_think)
        metrics = run_generation(model, tokenizer, prompt, max_new_tokens, use_chat_template=use_chat_template)
        score = score_task(task, str(metrics.get("generated_text", "")))
        rows.append(
            {
                "id": task["id"],
                "type": task["type"],
                "prompt": prompt,
                "answer": task["answer"],
                "generated_text": metrics.get("generated_text", ""),
                "score": score,
                "ttft_seconds": metrics.get("ttft_seconds"),
                "tokens_per_second": metrics.get("tokens_per_second"),
            }
        )
    return rows


def load_causal_lm(args: argparse.Namespace, dtype: Any):
    if args.loader == "hf":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
    if args.loader == "autoawq":
        from awq import AutoAWQForCausalLM

        device_map = {"": "cuda:0"} if str(args.device).startswith("cuda") else {"": args.device}
        return AutoAWQForCausalLM.from_quantized(
            args.model,
            trust_remote_code=True,
            fuse_layers=False,
            use_exllama=False,
            use_exllama_v2=False,
            safetensors=True,
            device_map=device_map,
            max_seq_len=args.max_seq_len,
        )
    raise ValueError(f"unsupported loader: {args.loader}")


def place_model(model: Any, device: Any) -> None:
    if hasattr(model, "model") and hasattr(model.model, "to"):
        model.model.to(device)
    elif hasattr(model, "to"):
        model.to(device)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Chat Task Benchmark",
        "",
        f"Model: `{result['model']}`",
        f"Tasks: `{result['task_count']}`",
        f"Task format: `{result.get('task_format', 'native')}`",
        f"Chat template: `{result['chat_template']}`",
        f"No-think prompt: `{result.get('no_think', False)}`",
        "",
        "## Aggregate",
        "",
        "| split | passes | accuracy | mean tok/s | mean TTFT s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "fused"):
        if name not in result:
            continue
        agg = result[name]["aggregate"]
        lines.append(
            f"| {name} | {agg['passes']} / {agg['tasks']} | {agg['accuracy']:.4f} | "
            f"{agg['mean_tokens_per_second']:.4f} | {agg['mean_ttft_seconds']:.6f} |"
        )
    lines.extend(["", "## Rows", "", "| split | id | type | passed | expected | generated |", "|---|---|---|---:|---|---|"])
    for name in ("baseline", "fused"):
        if name not in result:
            continue
        for row in result[name]["rows"]:
            generated = str(row["generated_text"]).replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(
                f"| {name} | `{row['id']}` | `{row['type']}` | {str(row['score']['passed']).lower()} | "
                f"`{row['answer']}` | `{generated}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chat model on deterministic JSONL tasks.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--task-format", choices=["native", "mmlu", "gsm8k", "ifeval"], default="native")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--loader", choices=["hf", "autoawq"], default="hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N tasks; 0 means all tasks.")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--no-think", action="store_true", help="Append /no_think once to each task prompt.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--package-summary", default="")
    parser.add_argument("--layers", default="")
    parser.add_argument("--suffixes", default="q_proj,k_proj,v_proj")
    parser.add_argument("--dense-roles", default="")
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--sync-mode", choices=["per_call", "end"], default="end")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from benchmark_esmp_fused_selected_rows import parse_suffixes
    from eval_esmp_module_reconstruction import load_package_modules, resolve_path, repo_root
    from measure_esmp_fused_qkv_generation import install_fused_qkv
    from measure_esmp_generation_latency import parse_layers

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tasks = apply_task_limit(load_tasks(Path(args.tasks_jsonl), task_format=args.task_format), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, trust_remote_code=True)
    model = load_causal_lm(args, dtype)
    model.eval()
    place_model(model, torch.device(args.device))

    baseline_rows = evaluate_rows(model, tokenizer, tasks, args.max_new_tokens, args.chat_template, args.no_think)
    result: dict[str, Any] = {
        "model": args.model,
        "loader": args.loader,
        "task_file": args.tasks_jsonl,
        "task_format": args.task_format,
        "task_count": len(tasks),
        "task_limit": args.limit,
        "chat_template": args.chat_template,
        "no_think": args.no_think,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {"aggregate": aggregate_rows(baseline_rows), "rows": baseline_rows},
    }

    if args.package_summary and args.layers:
        package_summary = resolve_path(args.package_summary, repo_root())
        runtimes = install_fused_qkv(
            model=model,
            package_modules=load_package_modules(package_summary),
            layers=sorted(parse_layers(args.layers) or set()),
            suffixes=parse_suffixes(args.suffixes),
            dense_roles=set(parse_suffixes(args.dense_roles)) if args.dense_roles else set(),
            device=torch.device(args.device),
            dtype=dtype,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            sync_mode=args.sync_mode,
        )
        fused_rows = evaluate_rows(model, tokenizer, tasks, args.max_new_tokens, args.chat_template, args.no_think)
        for runtime in runtimes:
            runtime.finalize_pending()
        result["fused"] = {"aggregate": aggregate_rows(fused_rows), "rows": fused_rows}

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "baseline": result["baseline"]["aggregate"], "fused": result.get("fused", {}).get("aggregate")}, indent=2))


if __name__ == "__main__":
    main()
