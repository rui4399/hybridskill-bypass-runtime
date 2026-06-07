#!/usr/bin/env python3
from __future__ import annotations

"""Gate public task subset evaluations.

This is a coverage and execution gate, not an accuracy leaderboard. It verifies
that real public-task JSONL evaluations ran under GPU guard and reached a
minimum task count.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_case_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(f"case must be LABEL=SUMMARY_JSON=GUARD_JSON: {spec}")
    label, summary_path, guard_path = parts
    return label.strip(), Path(summary_path), Path(guard_path)


def case_summary(label: str, summary_path: Path, guard_path: Path) -> dict[str, Any]:
    payload = load_json(summary_path)
    guard = load_json(guard_path)
    aggregate = payload.get("baseline", {}).get("aggregate", {}) or {}
    task_count = int(payload.get("task_count") or aggregate.get("tasks") or 0)
    passes = int(aggregate.get("passes") or 0)
    return {
        "label": label,
        "summary_path": str(summary_path),
        "guard_path": str(guard_path),
        "model": payload.get("model"),
        "task_format": payload.get("task_format"),
        "task_count": task_count,
        "passes": passes,
        "accuracy": (passes / task_count) if task_count else 0.0,
        "mean_tokens_per_second": float(aggregate.get("mean_tokens_per_second") or 0.0),
        "mean_ttft_seconds": float(aggregate.get("mean_ttft_seconds") or 0.0),
        "guard_returncode": guard.get("returncode"),
        "killed_by_guard": bool(guard.get("killed_by_guard")),
        "killed_by_timeout": bool(guard.get("killed_by_timeout")),
        "guard_max_memory_used_ratio": float(guard.get("max_memory_used_ratio") or 0.0),
        "guard_max_memory_used_mib": guard.get("max_memory_used_mib"),
        "guard_memory_total_mib": guard.get("memory_total_mib"),
    }


def _formats(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def build_result(cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    total_tasks = sum(case["task_count"] for case in cases)
    total_passes = sum(case["passes"] for case in cases)
    formats = sorted({str(case.get("task_format") or "") for case in cases if case.get("task_format")})
    required_formats = _formats(args.require_formats)

    if len(cases) < args.min_cases:
        failures.append(f"case count {len(cases)} < required {args.min_cases}")
    if total_tasks < args.min_total_tasks:
        failures.append(f"total tasks {total_tasks} < required {args.min_total_tasks}")
    missing_formats = sorted(required_formats.difference(formats))
    if missing_formats:
        failures.append(f"missing required task formats: {missing_formats}")

    for case in cases:
        if case["guard_returncode"] != 0:
            failures.append(f"{case['label']}: guard returncode {case['guard_returncode']} != 0")
        if case["killed_by_guard"]:
            failures.append(f"{case['label']}: killed by GPU guard")
        if case["killed_by_timeout"]:
            failures.append(f"{case['label']}: killed by timeout")
        if case["guard_max_memory_used_ratio"] > args.max_memory_ratio:
            failures.append(
                f"{case['label']}: max memory ratio {case['guard_max_memory_used_ratio']:.4f} > {args.max_memory_ratio:.4f}"
            )
        if case["mean_tokens_per_second"] < args.min_mean_tokens_per_second:
            failures.append(
                f"{case['label']}: mean tok/s {case['mean_tokens_per_second']:.4f} < {args.min_mean_tokens_per_second:.4f}"
            )

    return {
        "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "passed": not failures,
        "summary": {
            "case_count": len(cases),
            "total_tasks": total_tasks,
            "total_passes": total_passes,
            "mean_accuracy": (total_passes / total_tasks) if total_tasks else 0.0,
            "task_formats": formats,
            "max_guard_vram_ratio": max((case["guard_max_memory_used_ratio"] for case in cases), default=0.0),
            "mean_tokens_per_second": mean([case["mean_tokens_per_second"] for case in cases]) if cases else 0.0,
            "mean_ttft_seconds": mean([case["mean_ttft_seconds"] for case in cases]) if cases else 0.0,
        },
        "cases": cases,
        "failures": failures,
        "claim_boundary": (
            "Valid claim: public MMLU/GSM8K subset evaluation ran under guard. "
            "Invalid claim: this is leaderboard-scale or SOTA quality evidence."
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Public Task Benchmark Gate",
        "",
        f"Date: `{result['date']}`",
        f"Status: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"Cases: `{summary['case_count']}`",
        f"Total tasks: `{summary['total_tasks']}`",
        f"Total passes: `{summary['total_passes']}`",
        f"Mean accuracy: `{summary['mean_accuracy']:.4f}`",
        f"Peak guard VRAM ratio: `{summary['max_guard_vram_ratio']:.4f}`",
        "",
        "## Cases",
        "",
        "| case | format | tasks | passes | accuracy | mean tok/s | mean TTFT s | VRAM ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| `{case['label']}` | `{case['task_format']}` | {case['task_count']} | {case['passes']} | "
            f"{case['accuracy']:.4f} | {case['mean_tokens_per_second']:.4f} | "
            f"{case['mean_ttft_seconds']:.6f} | {case['guard_max_memory_used_ratio']:.4f} |"
        )
    lines.extend(["", "## Failures", ""])
    if result["failures"]:
        for failure in result["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", f"- {result['claim_boundary']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate public task subset evaluations.")
    parser.add_argument("--case", action="append", required=True, help="LABEL=SUMMARY_JSON=GUARD_JSON")
    parser.add_argument("--min-cases", type=int, default=2)
    parser.add_argument("--min-total-tasks", type=int, default=100)
    parser.add_argument("--require-formats", default="mmlu,gsm8k")
    parser.add_argument("--max-memory-ratio", type=float, default=0.90)
    parser.add_argument("--min-mean-tokens-per-second", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    cases = [case_summary(label, summary_path, guard_path) for label, summary_path, guard_path in map(parse_case_spec, args.case)]
    result = build_result(cases, args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(json.dumps({"passed": result["passed"], "summary": result["summary"], "out_json": str(args.out_json)}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
