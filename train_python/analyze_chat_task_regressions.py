#!/usr/bin/env python3
from __future__ import annotations

"""Analyze baseline/fused pass-count flips in chat task benchmark outputs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_id(result: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    rows = result.get(split, {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"result missing {split}.rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["id"])
        if task_id in by_id:
            raise ValueError(f"duplicate task id in {split}: {task_id}")
        by_id[task_id] = row
    return by_id


def _passed(row: dict[str, Any]) -> bool:
    return bool(row.get("score", {}).get("passed"))


def _short(text: Any, limit: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _flip_record(task_id: str, base: dict[str, Any], fused: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "type": str(base.get("type", fused.get("type", ""))),
        "kind": kind,
        "expected": base.get("answer", fused.get("answer")),
        "baseline_text": _short(base.get("generated_text")),
        "fused_text": _short(fused.get("generated_text")),
        "baseline_score": base.get("score", {}),
        "fused_score": fused.get("score", {}),
    }


def analyze_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    baseline_rows = _rows_by_id(result, "baseline")
    fused_rows = _rows_by_id(result, "fused")
    if set(baseline_rows) != set(fused_rows):
        missing_fused = sorted(set(baseline_rows) - set(fused_rows))
        missing_base = sorted(set(fused_rows) - set(baseline_rows))
        raise ValueError(f"baseline/fused task ids differ; missing_fused={missing_fused}; missing_base={missing_base}")

    type_summary: dict[str, dict[str, int]] = {}
    regressions: list[dict[str, Any]] = []
    fixes: list[dict[str, Any]] = []

    for task_id in sorted(baseline_rows):
        base = baseline_rows[task_id]
        fused = fused_rows[task_id]
        task_type = str(base.get("type", fused.get("type", "")))
        bucket = type_summary.setdefault(
            task_type,
            {"tasks": 0, "baseline_passes": 0, "fused_passes": 0, "regressions": 0, "fixes": 0, "both_pass": 0, "both_fail": 0},
        )
        base_pass = _passed(base)
        fused_pass = _passed(fused)
        bucket["tasks"] += 1
        bucket["baseline_passes"] += int(base_pass)
        bucket["fused_passes"] += int(fused_pass)
        if base_pass and fused_pass:
            bucket["both_pass"] += 1
        elif not base_pass and not fused_pass:
            bucket["both_fail"] += 1
        elif base_pass and not fused_pass:
            bucket["regressions"] += 1
            regressions.append(_flip_record(task_id, base, fused, "regression"))
        else:
            bucket["fixes"] += 1
            fixes.append(_flip_record(task_id, base, fused, "fix"))

    base_agg = result.get("baseline", {}).get("aggregate", {})
    fused_agg = result.get("fused", {}).get("aggregate", {})
    baseline_passes = int(base_agg.get("passes", sum(item["baseline_passes"] for item in type_summary.values())))
    fused_passes = int(fused_agg.get("passes", sum(item["fused_passes"] for item in type_summary.values())))
    base_tps = float(base_agg.get("mean_tokens_per_second") or 0.0)
    fused_tps = float(fused_agg.get("mean_tokens_per_second") or 0.0)
    return {
        "label": label,
        "tasks": len(baseline_rows),
        "baseline_passes": baseline_passes,
        "fused_passes": fused_passes,
        "pass_delta": fused_passes - baseline_passes,
        "baseline_tps": base_tps,
        "fused_tps": fused_tps,
        "speedup": fused_tps / base_tps if base_tps > 0.0 else 0.0,
        "type_summary": dict(sorted(type_summary.items())),
        "regressions": regressions,
        "fixes": fixes,
    }


def parse_candidate(spec: str, root: Path) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"candidate must be LABEL=JSON: {spec}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    report = analyze_result(load_json(path), label.strip())
    report["path"] = str(path)
    return report


def build_report(candidates: list[str], root: Path) -> dict[str, Any]:
    reports = [parse_candidate(candidate, root) for candidate in candidates]
    return {
        "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "reports": reports,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Chat Task Regression Analysis",
        "",
        f"Date: `{report['date']}`",
        "",
        "This report compares baseline and fused rows from `eval_chat_task_benchmark.py` outputs. Regressions are rows where baseline passed and fused failed; fixes are the reverse.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | passes | pass delta | speed | regressions | fixes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["reports"]:
        lines.append(
            f"| `{item['label']}` | {item['baseline_passes']} -> {item['fused_passes']} / {item['tasks']} | "
            f"{item['pass_delta']} | {item['speedup']:.4f}x | {len(item['regressions'])} | {len(item['fixes'])} |"
        )

    for item in report["reports"]:
        lines.extend(["", f"## `{item['label']}` Type Summary", "", "| type | tasks | baseline pass | fused pass | regressions | fixes | both pass | both fail |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for task_type, summary in item["type_summary"].items():
            lines.append(
                f"| `{task_type}` | {summary['tasks']} | {summary['baseline_passes']} | {summary['fused_passes']} | "
                f"{summary['regressions']} | {summary['fixes']} | {summary['both_pass']} | {summary['both_fail']} |"
            )
        lines.extend(["", f"### `{item['label']}` Regressions", "", "| id | type | expected | baseline | fused |", "|---|---|---|---|---|"])
        if not item["regressions"]:
            lines.append("| n/a | n/a | n/a | n/a | n/a |")
        for row in item["regressions"]:
            lines.append(
                f"| `{row['id']}` | `{row['type']}` | `{_short(row['expected'], 80)}` | "
                f"`{_short(row['baseline_text'], 120)}` | `{_short(row['fused_text'], 120)}` |"
            )
        lines.extend(["", f"### `{item['label']}` Fixes", "", "| id | type | expected | baseline | fused |", "|---|---|---|---|---|"])
        if not item["fixes"]:
            lines.append("| n/a | n/a | n/a | n/a | n/a |")
        for row in item["fixes"]:
            lines.append(
                f"| `{row['id']}` | `{row['type']}` | `{_short(row['expected'], 80)}` | "
                f"`{_short(row['baseline_text'], 120)}` | `{_short(row['fused_text'], 120)}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze baseline/fused task pass regressions.")
    parser.add_argument("--candidate", action="append", required=True, help="LABEL=eval_json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)

    root = Path.cwd()
    report = build_report(args.candidate, root)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, report)
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "reports": len(report["reports"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
