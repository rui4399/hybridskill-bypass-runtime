#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate deterministic public-task JSONL files through Ollama.

This keeps public-task coverage moving when the local HF/torch runtime is not
available. It reuses the repository scoring logic from eval_chat_task_benchmark.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from eval_chat_task_benchmark import aggregate_rows, format_task_prompt, load_tasks, render_markdown, score_task


GenerateFn = Callable[[str, str, int, str], dict[str, Any]]


def run_ollama_generation(model: str, prompt: str, max_new_tokens: int, endpoint: str) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
        },
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []
    final: dict[str, Any] = {}
    try:
        with urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line.decode("utf-8"))
                text = str(item.get("response") or "")
                if text and first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(text)
                if item.get("done"):
                    final = item
                    break
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    end = time.perf_counter()

    generated_text = "".join(chunks)
    eval_count = int(final.get("eval_count") or 0)
    eval_duration_s = float(final.get("eval_duration") or 0.0) / 1_000_000_000.0
    tokens_per_second = (eval_count / eval_duration_s) if eval_count and eval_duration_s > 0 else 0.0
    return {
        "generated_text": generated_text,
        "ttft_seconds": (first_token_time - start) if first_token_time is not None else end - start,
        "elapsed_seconds": end - start,
        "tokens_per_second": tokens_per_second,
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration_s,
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_duration_seconds": float(final.get("prompt_eval_duration") or 0.0) / 1_000_000_000.0,
    }


def evaluate_rows(
    tasks: list[dict[str, Any]],
    *,
    model: str,
    max_new_tokens: int,
    endpoint: str,
    no_think: bool,
    generate_fn: GenerateFn = run_ollama_generation,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = format_task_prompt(str(task["prompt"]), no_think=no_think)
        metrics = generate_fn(model, prompt, max_new_tokens, endpoint)
        generated_text = str(metrics.get("generated_text", ""))
        rows.append(
            {
                "id": task["id"],
                "type": task["type"],
                "prompt": prompt,
                "answer": task["answer"],
                "generated_text": generated_text,
                "score": score_task(task, generated_text),
                "ttft_seconds": metrics.get("ttft_seconds"),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "ollama": {
                    "elapsed_seconds": metrics.get("elapsed_seconds"),
                    "eval_count": metrics.get("eval_count"),
                    "eval_duration_seconds": metrics.get("eval_duration_seconds"),
                    "prompt_eval_count": metrics.get("prompt_eval_count"),
                    "prompt_eval_duration_seconds": metrics.get("prompt_eval_duration_seconds"),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic JSONL tasks through Ollama.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--task-format", choices=["native", "mmlu", "gsm8k", "ifeval"], default="native")
    parser.add_argument("--model", default="huihui-qwen35-4b-pmra:latest")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks_jsonl), task_format=args.task_format)
    rows = evaluate_rows(
        tasks,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        endpoint=args.endpoint,
        no_think=args.no_think,
    )
    result = {
        "model": args.model,
        "backend": "ollama",
        "endpoint": args.endpoint,
        "task_file": args.tasks_jsonl,
        "task_format": args.task_format,
        "task_count": len(tasks),
        "chat_template": False,
        "no_think": args.no_think,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {"aggregate": aggregate_rows(rows), "rows": rows},
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "baseline": result["baseline"]["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
