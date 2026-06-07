#!/usr/bin/env python3
from __future__ import annotations

"""Gate chat-task regression analysis evidence.

This gate consumes the multi-candidate output produced by
`analyze_chat_task_regressions.py`. It is intentionally retention-oriented:
the fused candidate must keep enough deterministic task passes, avoid too many
baseline-pass -> fused-fail regressions, and stay within the GPU guard.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def find_report(analysis: dict[str, Any], label: str) -> dict[str, Any]:
    for report in analysis.get("reports", []) or []:
        if str(report.get("label")) == label:
            return report
    available = [str(report.get("label")) for report in analysis.get("reports", []) or []]
    raise ValueError(f"candidate {label!r} not found; available={available}")


def check_guard(guard: dict[str, Any] | None, args: argparse.Namespace) -> list[str]:
    if guard is None:
        return []
    failures: list[str] = []
    if int(guard.get("returncode", -1)) != 0:
        failures.append(f"guarded command returncode {guard.get('returncode')} != 0")
    if guard.get("killed_by_guard"):
        failures.append("guard killed command for memory")
    if guard.get("killed_by_timeout"):
        failures.append("guard killed command for timeout")
    ratio = finite_float(guard.get("max_memory_used_ratio"))
    if ratio is None:
        failures.append("guard missing max_memory_used_ratio")
    elif ratio > args.max_memory_ratio:
        failures.append(f"guard max memory ratio {ratio:.4f} > {args.max_memory_ratio:.4f}")
    return failures


def build_result(analysis: dict[str, Any], guard: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    reports = analysis.get("reports", []) or []
    if len(reports) < args.min_candidates:
        failures.append(f"candidate reports {len(reports)} < {args.min_candidates}")

    target = find_report(analysis, args.candidate)
    tasks = int(target.get("tasks", 0) or 0)
    baseline_passes = int(target.get("baseline_passes", 0) or 0)
    fused_passes = int(target.get("fused_passes", 0) or 0)
    pass_delta = int(target.get("pass_delta", fused_passes - baseline_passes) or 0)
    speedup = finite_float(target.get("speedup"))
    regressions = list(target.get("regressions", []) or [])
    fixes = list(target.get("fixes", []) or [])
    type_summary = target.get("type_summary", {}) or {}
    fused_accuracy = fused_passes / max(tasks, 1)
    baseline_accuracy = baseline_passes / max(tasks, 1)

    if tasks < args.min_tasks:
        failures.append(f"tasks {tasks} < {args.min_tasks}")
    if len(type_summary) < args.min_task_types:
        failures.append(f"task types {len(type_summary)} < {args.min_task_types}")
    if baseline_passes < args.min_baseline_passes:
        failures.append(f"baseline passes {baseline_passes} < {args.min_baseline_passes}")
    if fused_passes < args.min_fused_passes:
        failures.append(f"fused passes {fused_passes} < {args.min_fused_passes}")
    if fused_accuracy < args.min_fused_accuracy:
        failures.append(f"fused accuracy {fused_accuracy:.4f} < {args.min_fused_accuracy:.4f}")
    if pass_delta < -args.max_pass_loss:
        failures.append(f"pass delta {pass_delta} < -{args.max_pass_loss}")
    if len(regressions) > args.max_regressions:
        failures.append(f"regressions {len(regressions)} > {args.max_regressions}")
    if speedup is None or speedup < args.min_speedup:
        failures.append(f"speedup {speedup} < {args.min_speedup}")

    type_failures: list[str] = []
    for task_type, summary in sorted(type_summary.items()):
        regressions_for_type = int(summary.get("regressions", 0) or 0)
        fused_type_passes = int(summary.get("fused_passes", 0) or 0)
        if regressions_for_type > args.max_regressions_per_type:
            type_failures.append(f"{task_type}: regressions {regressions_for_type} > {args.max_regressions_per_type}")
        if args.require_type_fused_pass and fused_type_passes <= 0:
            type_failures.append(f"{task_type}: fused passes {fused_type_passes} <= 0")
    failures.extend(type_failures)
    failures.extend(check_guard(guard, args))

    return {
        "passed": not failures,
        "failures": failures,
        "summary": {
            "candidate": args.candidate,
            "tasks": tasks,
            "task_types": len(type_summary),
            "baseline_passes": baseline_passes,
            "fused_passes": fused_passes,
            "baseline_accuracy": baseline_accuracy,
            "fused_accuracy": fused_accuracy,
            "pass_delta": pass_delta,
            "regressions": len(regressions),
            "fixes": len(fixes),
            "speedup": speedup,
            "path": target.get("path"),
            "guard_max_memory_used_ratio": None if guard is None else guard.get("max_memory_used_ratio"),
            "guard_max_memory_used_mib": None if guard is None else guard.get("max_memory_used_mib"),
            "guard_memory_total_mib": None if guard is None else guard.get("memory_total_mib"),
        },
        "type_summary": type_summary,
        "regressions": regressions,
        "fixes": fixes,
        "thresholds": {
            "candidate": args.candidate,
            "min_candidates": args.min_candidates,
            "min_tasks": args.min_tasks,
            "min_task_types": args.min_task_types,
            "min_baseline_passes": args.min_baseline_passes,
            "min_fused_passes": args.min_fused_passes,
            "min_fused_accuracy": args.min_fused_accuracy,
            "max_pass_loss": args.max_pass_loss,
            "max_regressions": args.max_regressions,
            "max_regressions_per_type": args.max_regressions_per_type,
            "require_type_fused_pass": args.require_type_fused_pass,
            "min_speedup": args.min_speedup,
            "max_memory_ratio": args.max_memory_ratio,
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Chat Task Regression Gate",
        "",
        f"Status: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "## Summary",
        "",
        f"- candidate: `{summary['candidate']}`",
        f"- tasks: {summary['tasks']}",
        f"- task types: {summary['task_types']}",
        f"- baseline passes: {summary['baseline_passes']} ({fmt(summary['baseline_accuracy'])})",
        f"- fused passes: {summary['fused_passes']} ({fmt(summary['fused_accuracy'])})",
        f"- pass delta: {summary['pass_delta']}",
        f"- regressions / fixes: {summary['regressions']} / {summary['fixes']}",
        f"- mean speed ratio fused/baseline: {fmt(summary['speedup'])}x",
        f"- guard peak memory: {summary['guard_max_memory_used_mib']} / {summary['guard_memory_total_mib']} MiB ({fmt(summary['guard_max_memory_used_ratio'])})",
        "",
        "## By Task Type",
        "",
        "| type | tasks | baseline pass | fused pass | regressions | fixes | both pass | both fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_type, item in sorted(result["type_summary"].items()):
        lines.append(
            f"| `{task_type}` | {item.get('tasks')} | {item.get('baseline_passes')} | {item.get('fused_passes')} | "
            f"{item.get('regressions')} | {item.get('fixes')} | {item.get('both_pass')} | {item.get('both_fail')} |"
        )
    lines.extend(["", "## Regressions", "", "| id | type | expected | baseline | fused |", "|---|---|---|---|---|"])
    if not result["regressions"]:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    for row in result["regressions"]:
        expected = str(row.get("expected", "")).replace("|", "\\|")
        baseline = str(row.get("baseline_text", "")).replace("|", "\\|")
        fused = str(row.get("fused_text", "")).replace("|", "\\|")
        lines.append(f"| `{row.get('id')}` | `{row.get('type')}` | `{expected}` | `{baseline}` | `{fused}` |")
    lines.extend(["", "## Failures", ""])
    if result["failures"]:
        lines.extend(f"- {failure}" for failure in result["failures"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Valid claim: this candidate passes a deterministic 84-task stress-retention gate under the configured regression budget.",
            "- Invalid claim: this is a broad benchmark replacement for MMLU/GSM8K/IFEval or evidence of full quality preservation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate chat-task regression analysis.")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--guard-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/real_system_packer_2026-06-05/chat_task_regression_gate.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/real_system_packer_2026-06-05/CHAT_TASK_REGRESSION_GATE.md"))
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--min-tasks", type=int, default=1)
    parser.add_argument("--min-task-types", type=int, default=1)
    parser.add_argument("--min-baseline-passes", type=int, default=0)
    parser.add_argument("--min-fused-passes", type=int, default=0)
    parser.add_argument("--min-fused-accuracy", type=float, default=0.0)
    parser.add_argument("--max-pass-loss", type=int, default=0)
    parser.add_argument("--max-regressions", type=int, default=0)
    parser.add_argument("--max-regressions-per-type", type=int, default=0)
    parser.add_argument("--require-type-fused-pass", action="store_true")
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument("--max-memory-ratio", type=float, default=0.90)
    args = parser.parse_args()

    guard = load_json(args.guard_json) if args.guard_json else None
    result = build_result(load_json(args.analysis_json), guard, args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(json.dumps({"passed": result["passed"], "failures": result["failures"], "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
